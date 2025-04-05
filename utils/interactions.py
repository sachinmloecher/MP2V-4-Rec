from dataclasses import dataclass
from typing import Optional, Union

from utils.sql_vendor.query_stuck import QueryStuck, querystuck


@dataclass
class Interactions:
    window_start_date: str
    window_end_date: str
    partition_date: str
    plays_audited_src: Union[QueryStuck, str] = "sc-plays.views_daily.plays_audited"
    favoritings_src: Union[QueryStuck, str] = "sc-db-dumps.liebling.favoritings"
    reposts_src: Union[QueryStuck, str] = "sc-db-dumps.reposts.track_reposts"
    track_attributes_src: Union[QueryStuck, str] = "sc-corpus.views_daily.track_attributes"
    pii_users_lookup: Union[QueryStuck, str] = "sc-data-pii.hash_lookup_tables.users"
    pii_tracks_lookup: Union[QueryStuck, str] = "sc-data-pii.hash_lookup_tables.tracks"

    max_daily_interactions_allowed: Optional[int] = 300
    require_min_seconds_of_playback: Union[int, float] = 30
    require_musiio_data_on_tracks: bool = False

    @querystuck
    def variables(self):
        return f"""
        SELECT
        -- Start date of window of interactions to consider
        DATE('{self.window_start_date}') as start_date,

        -- End date of window of interactions to consider
        DATE('{self.window_end_date}') as end_date,

        -- Partition Date used for tables partitioned daily
        DATE('{self.partition_date}') as partition_date,

        -- Minimum playback duration to consider a valid play
        {self.require_min_seconds_of_playback} * 1000 as min_duration_millis,

        -- Maximum number of interactions allowed per day
        {self.max_daily_interactions_allowed} as max_interactions
    """

    @querystuck
    def plays_audited_query(self, plays_audited_src, variables):
        """Events whereby a user played a track that was not recommended to them / on auto-play."""

        return f"""
        SELECT
            user                  AS user,
            track                 AS track,
            1                     AS interaction_type,
            MAX(received_time) AS interaction_time

        FROM {plays_audited_src} AS pa, {variables} as vars
        WHERE trusted
          AND duration >= vars.min_duration_millis
          AND track_owner != user
          AND user IS NOT NULL
          AND date_id BETWEEN vars.start_date AND vars.end_date

          /* Filter out recommended/autoplay events */
          AND NOT (
            origin.source IN (
                'personalized-tracks', 'weekly', 'new-for-you', 'your-playback',
                'personal-recommended', 'picks-for-you', 'hidden-gems', 'recommender'
            )
            OR origin.source LIKE ('%station%')
            OR origin.source LIKE ('%similar-to%')
          )
        GROUP BY user, track
    """

    @querystuck
    def favoritings_query(self, favoritings_src, variables):
        return f"""
          SELECT
            user,
            track,
            2 AS interaction_type,
            created_at AS interaction_time
          FROM {favoritings_src}, {variables} AS vars
          WHERE _PARTITIONDATE = vars.partition_date
          AND DATE(created_at) BETWEEN vars.start_date and vars.end_date
        """

    @querystuck
    def track_reposts_query(self, reposts_src, variables):
        return f"""
        SELECT
          user,
          track,
          3 AS interaction_type,
          created_at AS interaction_time
        FROM {reposts_src}, {variables} AS vars
        WHERE _PARTITIONDATE = vars.partition_date
        AND DATE(created_at) BETWEEN vars.start_date and vars.end_date
        """

    @querystuck
    def user_track_interactions(self, plays_audited, favoritings, track_reposts):
        return f"""
        SELECT
          user,
          track,
          interaction_type,
          interaction_time
        FROM (
          SELECT * FROM {plays_audited}
          UNION ALL
          SELECT * FROM {favoritings}
          UNION ALL
          SELECT * FROM {track_reposts}
        )
        """

    @querystuck
    def filter_max_daily_user_track_interactions(
        self,
        user_track_interactions,
        variables,
    ):
        return f"""
          SELECT
           user,
           track,
           interaction_type,
           interaction_time
          FROM {user_track_interactions}, {variables} as vars
          WHERE user NOT IN (
            SELECT DISTINCT user
            FROM {user_track_interactions}
            GROUP BY user, DATE(interaction_time)
            HAVING COUNT(1) > vars.max_interactions)
        """

    @querystuck
    def filter_tracks_without_musiio_data(
        self,
        user_track_interactions,
        track_attributes,
        variables,
    ):
        return f"""
          SELECT
           i.user,
           i.track,
           i.interaction_type,
           i.interaction_time
          FROM {variables} as vars, {user_track_interactions} i
          JOIN {track_attributes} t
          ON t.track = i.track
          WHERE t.date_id = vars.partition_date
         """

    @querystuck
    def resolve_and_rank_interactions(
        self,
        user_track_interactions,
        user_lookup,
        track_lookup,
        variables,
    ):
        return f"""
          SELECT
            u.user_id          AS user_id,
            t.track_id         AS track_id,
            i.interaction_type AS interaction_type,
            i.interaction_time AS interaction_time,
            RANK() OVER (PARTITION BY user_id ORDER BY i.interaction_time)
            AS interaction_rank

          FROM {variables} as vars, {user_track_interactions} AS i
          JOIN {user_lookup} AS u USING (user)
          JOIN {track_lookup} AS t USING (track)
        """

    def full_query(self):
        """Build the query that generates the dataset."""
        variables = self.variables()

        play_events = self.plays_audited_query(
            plays_audited_src=self.plays_audited_src,
            variables=variables,
        )

        likes = self.favoritings_query(
            favoritings_src=self.favoritings_src,
            variables=variables,
        )

        reposts = self.track_reposts_query(
            reposts_src=self.reposts_src,
            variables=variables,
        )

        hashed_interactions = self.user_track_interactions(
            plays_audited=play_events,
            favoritings=likes,
            track_reposts=reposts,
        )

        hashed_interactions = self.filter_max_daily_user_track_interactions(
            user_track_interactions=hashed_interactions,
            variables=variables,
        )

        if self.require_musiio_data_on_tracks:
            hashed_interactions = self.filter_tracks_without_musiio_data(
                user_track_interactions=hashed_interactions,
                track_attributes=self.track_attributes_src,
                variables=variables,
            )

        return self.resolve_and_rank_interactions(
            user_track_interactions=hashed_interactions,
            user_lookup=self.pii_users_lookup,
            track_lookup=self.pii_tracks_lookup,
            variables=variables,
        )