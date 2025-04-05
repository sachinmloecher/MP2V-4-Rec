
# pylint: skip-file
"""
This is QueryStuck, a tiny library for composable sql.

# So, what is it?

A QueryStuck is conceptually the same thing as SELECT statement. It's a piece of SQL ready to run.
However, a QueryStuck holds a bit more information than just a SQL string, which gives it magic powers of composability,
and more!

# How do I create one?

The easiest way make a QueryStuck is to make a function that returns a bit of SQL, and to decorate it with `querystuck`.
Here is an example:
```
@querystuck
def clicks_on_date(date, click="`sc-data-events.events.click`"):
    return f'''SELECT * FROM {click} WHERE _PARTITIONDATE = '{date}' '''

@querystuck
def count_clicks(click):
    return f'''SELECT COUNT(*) FROM {click} '''
```

Some arguments of this function will be inputs (like `click`), which can be other tables in BigQuery (as is the case for
the default argument).
However, we could also another piece of SQL here, like `(SELECT '2020-10-27' as _PARTITION_DATE`), for testing purpose.
We could even pass another piece of query that we have assembled before, for example to filter some events before we do
the counting.

This gives us the basis of composability: we create bits of queries that can take other bits of queries as inputs.


# So, what does QueryStuck gives me?

You may wonder why we need a library to support this pattern. We could very well build our queries using this technique,
and be happy with it!
QueryStuck gives us a few things beyond simply assembling our queries. Let's illustrate by using the `count_clicks`
example.

When we call the `count_clicks` function, we get a QueryStuck instead of a raw SQL string.

```
>>> query = count_clicks(click=clicks_on_date(date='2020-10-27'))
QueryStuck(name='count_clicks', ...)

```
There are a few things we can do with the QueryStuck.
First of all, we can get the SQL query we would have had in the first place.

```
>>> query.render_inline()
SELECT COUNT(*) FROM (SELECT * FROM `sc-data-events.events.click` WHERE _PARTITIONDATE = '2020-10-27' )
```

We can also use the extra power of QueryStuck to generate a piece of SQL that flattens all inputs, the same way a
human would have written it.

```
>>> query.render_using_with()
WITH clicks_on_date AS (SELECT * FROM `sc-data-events.events.click` WHERE _PARTITIONDATE = '2020-10-27' ),
count_clicks AS (SELECT COUNT(*) FROM clicks_on_date )
SELECT * FROM count_clicks
```

And finally, we can diplay a tree of the query, to understand how the various parts connect together.
```
>>> query.print_tree()
count_clicks()
├──clicks_on_date(date='2020-10-27')
```

If you're curious how it works, take a look at the doc for the `QueryStuck` class.


# How do I use that for testing?

Since we can pass whatever we want as an input, we can totally pass a piece of SQL that returns some rows from scratch
(like `SELECT 1 AS x, 2 AS y`)!
But to make things extra easy, we have some support for python dataclasses with the `selectable` function, so you can
do this:

```
@dataclass
class Click:
    ts: int
    _PARTITIONDATE: str

>>> a_click = selectable(Click(ts=0, _PARTITIONDATE='2020-10-27'))
>>> query = count_clicks(click=clicks_on_date(date='2020-10-27', click=a_click))
>>> query.print_tree()
count_clicks()
├──clicks_on_date(date='2020-10-27')
│   ├──Click()
>>> print(query.render_using_with())
WITH Click AS (SELECT
  0 AS ts,
  '2020-10-27' AS _PARTITIONDATE),
clicks_on_date AS (SELECT * FROM Click WHERE _PARTITIONDATE = '2020-10-27' ),
count_clicks AS (SELECT COUNT(*) FROM clicks_on_date )
SELECT * FROM count_clicks
```

You can then run this in BigQuery, and do assertions on the results!
(More documentation for this once we package it properly ;
 for now, you can find an example in test_clean_loads_and_clicks.py)
"""
from dataclasses import dataclass, field
from datetime import date
from functools import wraps
from typing import Callable, Dict

from similar_sounds.utils.sql_vendor.dataproxy import (
    as_sql,
    extract_name,
    serialized_value,
)


@dataclass
class QueryStuck:
    """
    Essentially, a QueryStuck contains everything you need to generate SQL.
    However, we keep all the ingredients needed to generate the SQL, instead of the SQL itself, so we can use it in
    different ways.

    What we store is:
    - The `name` of the function that would be used to generate it
    - `fn`, the actual function that generates the SQL
    - `stuck_inputs`, the QueryStuck arguments the function has been called with
    - `args`, the other arguments the function has been called with

    This allow us to do different things:
    - Call the function with the arguments untouched, to get some ugly SQL (That's `render_inline`)
    - Collect all the QueryStuck recursively (ie, including the stucks that are in other stucks),
        and then render them all as a nice list in a `WITH` statement (which is how a human writes these things). That's
        `render_using_with`, and that's the one you probably want.
    - Walk all the QueryStucks recursively, just to get their names and arguments, and render it as a since tree. That's
        `print_tree`
    """

    name: str
    fn: Callable = field(repr=False)
    args: dict = field(repr=False)
    stuck_inputs: Dict[str, "QueryStuck"] = field(repr=False)

    def print_tree(self):
        self._print_tree(level=0)

    def _print_tree(self, level):
        continuation_line = "│   " * (level - 1)
        fork_line = "├──" if level > 0 else ""
        args = ", ".join(f"{k}={repr(v)}" for k, v in self.args.items())
        print(f"{continuation_line}{fork_line}{self.name}({args})")
        for dep in self.stuck_inputs.values():
            dep._print_tree(level + 1)

    def render_inline(self):
        rendered_inputs = {k: f"({v.render_inline()})" for k, v in self.stuck_inputs.items()}
        rendered_inputs.update(self.args)
        return self.fn(**rendered_inputs)

    def render_using_with(self, *, partition_field=None, partition_id=None):
        stucks: Dict[str, QueryStuck] = dict()
        self._collect_stucks(stucks)
        withs = ",\n".join(
            f"{k} AS ({v._render_in_with()})"
            for k, v in stucks.items()  # fmt: skip
        )
        if partition_field and partition_id:
            fields = f"{partition_id} AS {partition_field}, *"
        else:
            fields = "*"
        return (
            f"WITH\n"
            f"{withs}\n"
            f"SELECT {fields}\n"
            f"FROM {self.name}"  # fmt: skip
        )

    def _collect_stucks(self, output):
        if self.name in output.keys():
            assert self == output[self.name]
        else:
            for name, stuck in self.stuck_inputs.items():
                stuck._collect_stucks(output)
            output[self.name] = self

    def _render_in_with(self):
        input_args = {k: v.name for k, v in self.stuck_inputs.items()}
        return self.fn(**{**self.args, **input_args})

    def child(self, name):
        stucks = dict()
        self._collect_stucks(stucks)
        return stucks[name]

    @staticmethod
    def make(name, fn, **kwargs):
        args = dict()
        stuck_inputs = dict()
        for k, v in kwargs.items():
            if isinstance(v, QueryStuck):
                stuck_inputs[k] = v
            else:
                args[k] = v
        return QueryStuck(name=name, fn=fn, args=args, stuck_inputs=stuck_inputs)


def querystuck(meth):
    """Same as the querystuck decorator, but for methods"""

    @wraps(meth)
    def wrapper(self, **kwargs):
        def fn(**kwg):
            return meth(self, **kwg)

        return QueryStuck.make(name=meth.__name__, fn=fn, **kwargs)

    return wrapper


def selectable(value):
    """Make a QueryStuck from any dataclass value.

    You'll get a piece of SQL of the form `SELECT 1 AS x, 2 AS y`, with your data marshalled as SQL.
    """
    return QueryStuck(
        name=extract_name(value),
        fn=lambda: as_sql(value),
        args={},
        stuck_inputs={},
    )


def print_and_execute(bigquery, query):
    """Print the tree of the query, and execute it with the given bigquery connector."""
    query.print_tree()
    rendered_query = query.render_using_with()
    return [dict(row.items()) for row in bigquery.query(rendered_query).result()]
