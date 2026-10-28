"""
Tests for time zone aware ``TimeSeries``.

The time zone of a time zone aware time index is kept through creation,
serialization, and the operations that generate or extend time indexes.
Calendar frequencies (daily and longer) step on the local wall clock across
daylight saving time (DST) changes.
"""

import datetime
import json
import zoneinfo

import numpy as np
import pandas as pd
import pytest
from dateutil import tz as dateutil_tz

from darts import TimeSeries
from darts.models import LinearRegressionModel, NaiveSeasonal
from darts.utils import timeseries_generation as tg

# one zone per kind of time zone object: IANA string, `zoneinfo`, `dateutil`,
# fixed UTC offset, and UTC itself
ZONES = {
    "amsterdam": "Europe/Amsterdam",
    "new_york": zoneinfo.ZoneInfo("America/New_York"),
    "dateutil": dateutil_tz.gettz("Europe/Amsterdam"),
    "fixed": datetime.timezone(datetime.timedelta(hours=5, minutes=30)),
    "utc": datetime.UTC,
}
DST_ZONES = ["amsterdam", "new_york", "dateutil"]

# zones that change their clocks at midnight: the day of the change has no
# midnight (spring forward) or two (fall back)
MIDNIGHT_CHANGES = {
    "havana_spring": ("America/Havana", "2024-02-24"),
    "havana_autumn": ("America/Havana", "2024-10-24"),
    "santiago_spring": ("America/Santiago", "2024-08-29"),
}


def make_times(zone, kind="hourly", name="time"):
    """An index that crosses the autumn DST change of its zone."""
    day = pd.Timestamp("2024-11-02" if zone == "new_york" else "2024-10-26")
    if kind == "hourly":
        start = day + pd.Timedelta(hours=18)
        return pd.date_range(start, periods=60, freq="h", tz=ZONES[zone], name=name)
    if kind == "spring":
        day = pd.Timestamp("2024-03-09" if zone == "new_york" else "2024-03-30")
        start = day + pd.Timedelta(hours=18)
        return pd.date_range(start, periods=60, freq="h", tz=ZONES[zone], name=name)
    start = day - pd.Timedelta(days=15)
    return pd.date_range(start, periods=40, freq="D", tz=ZONES[zone], name=name)


def make_series(zone, kind="hourly"):
    times = make_times(zone, kind)
    values = np.sin(np.arange(len(times)) / 5.0) + np.arange(len(times)) / 10.0
    return TimeSeries.from_times_and_values(times, values)


def assert_same_times(actual, expected):
    pd.testing.assert_index_equal(actual, expected, exact=False, check_names=False)
    assert str(actual.dtype) == str(expected.dtype)
    assert actual.freq == expected.freq


def utc_wall_times(times):
    """The same instants as a time zone naive index in UTC (the existing
    convention for time zone naive indexes)."""
    return pd.DatetimeIndex(times.tz_convert("UTC").tz_localize(None), freq=times.freq)


def reference_attribute(times, attribute, tz, **kwargs):
    """The expected attribute values, computed through the already tested time
    zone naive path."""
    return tg.datetime_attribute_timeseries(
        utc_wall_times(times),
        attribute,
        tz=tz if tz is not None else times.tz,
        **kwargs,
    ).values()


def local_days(case, periods=30):
    zone, start = MIDNIGHT_CHANGES[case]
    # the day of the change starts at its first existing moment; a repeated
    # midnight counts at its first occurrence
    return pd.date_range(
        start,
        periods=periods,
        freq="D",
        tz=zone,
        nonexistent="shift_forward",
        ambiguous=True,
        name="time",
    )


class TestTimeSeriesCreationWithTimeZone:
    @staticmethod
    def create(creator, times, values):
        frame = pd.DataFrame({"time": times, "value": values})
        if creator == "times_and_values":
            return [TimeSeries.from_times_and_values(times, values)]
        if creator == "pd_series":
            return [TimeSeries.from_series(pd.Series(values, index=times))]
        if creator == "dataframe_index":
            return [TimeSeries.from_dataframe(frame.set_index("time"))]
        if creator == "dataframe_time_col":
            return [TimeSeries.from_dataframe(frame, time_col="time")]
        groups = pd.concat([
            frame.assign(group="a"),
            frame.assign(group="b", value=values + 1.0),
        ])
        return TimeSeries.from_group_dataframe(
            groups, group_cols="group", time_col="time", value_cols="value"
        )

    @pytest.mark.parametrize("zone", list(ZONES))
    @pytest.mark.parametrize("kind", ["hourly", "daily"])
    @pytest.mark.parametrize(
        "creator",
        [
            "times_and_values",
            "pd_series",
            "dataframe_index",
            "dataframe_time_col",
            "group_dataframe",
        ],
    )
    def test_creation_keeps_time_zone(self, zone, kind, creator):
        times = make_times(zone, kind)
        values = np.arange(len(times), dtype=float)
        created = self.create(creator, times, values)
        for series in created:
            assert_same_times(series.time_index, times)
            assert series.time_index.tz is not None
        np.testing.assert_allclose(created[0].values()[:, 0], values)

    @pytest.mark.parametrize("zone", DST_ZONES)
    def test_spring_forward_hourly_series(self, zone):
        times = make_times(zone, "spring")
        df = pd.DataFrame({"time": times, "v": 1.0})
        series = TimeSeries.from_dataframe(df, time_col="time")
        assert_same_times(series.time_index, times)
        later = series.append_values(np.ones(3))
        expected_tail = pd.date_range(times[-1], periods=4, freq="h")
        assert_same_times(later.time_index[-4:], expected_tail)

    @pytest.mark.parametrize("zone", ["amsterdam", "new_york", "dateutil", "fixed"])
    @pytest.mark.parametrize("source", ["csv", "text_column"])
    def test_times_with_utc_offsets_load_as_utc(self, zone, source, tmp_path):
        # timestamps written with UTC offsets (e.g. in a CSV file) only
        # identify the instant, not the zone; they are read as UTC
        series = make_series(zone)
        if source == "csv":
            path = tmp_path / "series.csv"
            series.to_csv(path)
            loaded = TimeSeries.from_csv(path, time_col="time")
        else:
            frame = pd.DataFrame({
                "time": [str(t) for t in series.time_index],
                "v": series.values()[:, 0],
            })
            loaded = TimeSeries.from_dataframe(frame, time_col="time")
        assert_same_times(loaded.time_index, series.time_index.tz_convert("UTC"))
        np.testing.assert_allclose(loaded.values(), series.values())


class TestTimeSeriesSerializationWithTimeZone:
    @pytest.mark.parametrize("zone", list(ZONES))
    @pytest.mark.parametrize("kind", ["hourly", "daily"])
    def test_json_round_trip_keeps_time_zone(self, zone, kind):
        series = make_series(zone, kind)
        loaded = TimeSeries.from_json(series.to_json())
        assert_same_times(loaded.time_index, make_times(zone, kind))
        np.testing.assert_allclose(loaded.values(), series.values())
        # stepping on after loading follows the time zone, not a frozen UTC offset
        assert_same_times(
            loaded.append_values(np.ones(30)).time_index,
            series.append_values(np.ones(30)).time_index,
        )
        # the JSON format of time zone naive series is unchanged
        naive_times = series.time_index.tz_localize(None).drop_duplicates()[:10]
        naive = TimeSeries.from_times_and_values(naive_times, np.ones(10))
        assert TimeSeries.from_json(naive.to_json()) == naive
        assert json.loads(naive.to_json())["time_start"] == str(naive.start_time())

    @pytest.mark.parametrize("zone", list(ZONES))
    def test_pickle_round_trip_keeps_time_zone(self, zone, tmp_path):
        series = make_series(zone)
        # populate the cached boundary timestamps before pickling
        series.start_time(), series.end_time()
        path = str(tmp_path / "series.pkl")
        series.to_pickle(path)
        loaded = TimeSeries.from_pickle(path)
        assert_same_times(loaded.time_index, make_times(zone))
        assert loaded.start_time() == series.start_time()
        assert loaded.end_time() == series.end_time()

    @pytest.mark.parametrize("zone", list(ZONES))
    @pytest.mark.parametrize("model_kind", ["regression", "naive"])
    def test_trained_model_saves_and_loads_safely(self, zone, model_kind, tmp_path):
        # models trained on time zone aware series load with the default safe
        # loading, which does not allow pickling time zone objects directly
        series = make_series(zone)
        if model_kind == "regression":
            model = LinearRegressionModel(lags=3)
        else:
            model = NaiveSeasonal(K=2)
        model.fit(series[:40])
        expected = model.predict(10)
        path = str(tmp_path / "model.pkl")
        model.save(path)
        loaded = type(model).load(path)
        forecast = loaded.predict(10)
        assert_same_times(forecast.time_index, make_times(zone)[40:50])
        np.testing.assert_allclose(forecast.values(), expected.values())


class TestDatetimeAttributesWithTimeZone:
    @pytest.mark.parametrize("zone", list(ZONES))
    @pytest.mark.parametrize("attribute", ["hour", "dayofweek"])
    @pytest.mark.parametrize("tz", [None, "Asia/Kolkata"])
    def test_datetime_attributes_in_local_time(self, zone, attribute, tz):
        times = make_times(zone)
        expected = reference_attribute(times, attribute, tz)
        result = tg.datetime_attribute_timeseries(times, attribute, tz=tz)
        assert_same_times(result.time_index, times)
        np.testing.assert_allclose(result.values(), expected)
        stacked = make_series(zone).add_datetime_attribute(attribute, tz=tz)
        assert_same_times(stacked.time_index, times)
        np.testing.assert_allclose(stacked.values()[:, 1:], expected)
        for encoding in ["one_hot", "cyclic"]:
            result = tg.datetime_attribute_timeseries(
                times, attribute, tz=tz, **{encoding: True}
            )
            np.testing.assert_allclose(
                result.values(),
                reference_attribute(times, attribute, tz, **{encoding: True}),
            )

    @pytest.mark.parametrize("zone", ["amsterdam", "fixed"])
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_holidays_in_local_time(self, zone, tz):
        times = pd.date_range("2024-12-24 18:00", periods=60, freq="h", tz=ZONES[zone])
        result = tg.holidays_timeseries(times, "NL", tz=tz)
        assert_same_times(result.time_index, times)
        expected = tg.holidays_timeseries(
            utc_wall_times(times), "NL", tz=tz if tz is not None else times.tz
        ).values()
        np.testing.assert_allclose(result.values(), expected)
        series = TimeSeries.from_times_and_values(times, np.ones(len(times)))
        added = series.add_holidays("NL", tz=tz)
        np.testing.assert_allclose(added.values()[:, 1:], expected)
        extended = tg.holidays_timeseries(times, "NL", tz=tz, add_length=24)
        assert_same_times(
            extended.time_index, pd.date_range(times[0], periods=84, freq="h")
        )

    @pytest.mark.parametrize("zone", ["amsterdam", "new_york", "fixed"])
    @pytest.mark.parametrize(
        "encoder,tz",
        [
            ("future_attribute", None),
            ("future_attribute", "UTC"),
            ("future_cyclic", None),
            ("past_attribute", None),
        ],
    )
    def test_model_encoders_in_local_time(self, zone, encoder, tz):
        configs = {
            "future_attribute": (
                {"datetime_attribute": {"future": ["hour"]}},
                "future",
                "hour",
                {},
            ),
            "future_cyclic": (
                {"cyclic": {"future": ["hour"]}},
                "future",
                "hour",
                {"cyclic": True},
            ),
            "past_attribute": (
                {"datetime_attribute": {"past": ["dayofweek"]}},
                "past",
                "dayofweek",
                {},
            ),
        }
        config, temporal, attribute, kwargs = configs[encoder]
        add_encoders = dict(config, tz=tz) if tz is not None else dict(config)
        if temporal == "future":
            lags = {"lags_future_covariates": [0, 1]}
        else:
            lags = {"lags_past_covariates": [-1]}
        series = make_series(zone)
        model = LinearRegressionModel(lags=4, add_encoders=add_encoders, **lags)
        model.fit(series[:20])
        forecast = model.predict(24)
        assert_same_times(forecast.time_index, series.time_index[20:44])
        past, future = model.encoders.encode_inference(n=24, target=series[:20])
        covariates = future if temporal == "future" else past
        assert covariates.time_index.tz is not None
        expected = reference_attribute(covariates.time_index, attribute, tz, **kwargs)
        np.testing.assert_allclose(covariates.values(), expected)
        forecasts = model.historical_forecasts(
            series, start=30, forecast_horizon=2, retrain=False
        )
        assert_same_times(forecasts.time_index, series.time_index[31:])


class TestNaiveTimestampsAreLocalTime:
    # operations receiving a time zone naive timestamp interpret it as wall
    # time in the time zone of the series; aware timestamps compare by instant
    POINT_OPERATIONS = {
        "get_index_at_point": (
            lambda s, p: s.get_index_at_point(p),
            lambda t, p: t.searchsorted(p),
        ),
        "split_after": (
            lambda s, p: len(s.split_after(p)[0]),
            lambda t, p: t.searchsorted(p, side="right"),
        ),
        "split_before": (
            lambda s, p: len(s.split_before(p)[0]),
            lambda t, p: t.searchsorted(p),
        ),
        "drop_after": (
            lambda s, p: len(s.drop_after(p)),
            lambda t, p: t.searchsorted(p),
        ),
        "drop_before": (
            lambda s, p: len(s.drop_before(p)),
            lambda t, p: len(t) - t.searchsorted(p, side="right"),
        ),
        "slice": (
            lambda s, p: len(s.slice(p, p + pd.Timedelta(hours=5))),
            lambda t, p: int(((t >= p) & (t <= p + pd.Timedelta(hours=5))).sum()),
        ),
        "slice_n_points_after": (
            lambda s, p: s.slice_n_points_after(p, 3).start_time(),
            lambda t, p: p,
        ),
        "slice_n_points_before": (
            lambda s, p: s.slice_n_points_before(p, 3).end_time(),
            lambda t, p: p,
        ),
        "is_within_range": (lambda s, p: s.is_within_range(p), lambda t, p: True),
        "contains": (lambda s, p: p in s, lambda t, p: True),
        "getitem": (lambda s, p: s[p].start_time(), lambda t, p: p),
    }

    @pytest.mark.parametrize("zone", ["amsterdam", "new_york", "fixed"])
    @pytest.mark.parametrize("operation", list(POINT_OPERATIONS))
    def test_naive_points_are_local_time(self, zone, operation):
        series = make_series(zone)
        times = series.time_index
        call, reference = self.POINT_OPERATIONS[operation]
        # a wall time after the DST change of the zone
        day = "2024-11-03" if zone == "new_york" else "2024-10-27"
        point = pd.Timestamp(f"{day} 05:00")
        local = point.tz_localize(times.tz)
        assert call(series, point) == reference(times, local)
        # time zone aware points compare by instant
        assert call(series, local.tz_convert("UTC")) == reference(times, local)

    @pytest.mark.parametrize("method", ["historical_forecasts", "backtest"])
    def test_naive_start_is_local_time_per_series(self, method):
        # a naive `start` is the wall time in each series' own time zone
        first = TimeSeries.from_times_and_values(
            pd.date_range(
                "2024-10-26 18:00", periods=60, freq="h", tz="Europe/Amsterdam"
            ),
            np.arange(60.0),
        )
        second = TimeSeries.from_times_and_values(
            pd.date_range(
                "2024-10-26 18:00", periods=60, freq="h", tz="America/New_York"
            ),
            np.arange(60.0) * 2,
        )
        start = pd.Timestamp("2024-10-27 06:00")
        model = LinearRegressionModel(lags=3).fit([first, second])
        kwargs = dict(start=start, forecast_horizon=1, retrain=False)
        if method == "historical_forecasts":
            results = model.historical_forecasts([first, second], **kwargs)
        else:
            errors = model.backtest([first, second], **kwargs)
            expected = [
                model.backtest(
                    s,
                    start=start.tz_localize(s.time_index.tz),
                    forecast_horizon=1,
                    retrain=False,
                )
                for s in (first, second)
            ]
            np.testing.assert_allclose(errors, expected)
            results = model.historical_forecasts([first, second], **kwargs)
        for result, series in zip(results, [first, second]):
            local = start.tz_localize(series.time_index.tz)
            expected_times = series.time_index[series.time_index >= local]
            assert_same_times(result.time_index, expected_times)


class TestCalendarStepsAcrossMidnightChanges:
    # daily steps follow the local calendar also in zones that change their
    # clocks at midnight: the day of the change starts at its first existing
    # moment, and a repeated midnight counts once
    @staticmethod
    def run_daily_path(path, days):
        head = TimeSeries.from_times_and_values(days[:15], np.arange(15.0))
        if path == "times_and_values":
            return TimeSeries.from_times_and_values(days, np.arange(30.0)).time_index
        if path == "time_col":
            frame = pd.DataFrame({"time": days, "v": 1.0})
            return TimeSeries.from_dataframe(frame, time_col="time").time_index
        if path == "append_values":
            return head.append_values(np.ones(15)).time_index
        if path == "append":
            tail = TimeSeries.from_times_and_values(days[15:], np.ones(15))
            return head.append(tail).time_index
        if path == "shift":
            return head.shift(15).time_index
        if path == "json":
            series = TimeSeries.from_times_and_values(days, np.arange(30.0))
            return TimeSeries.from_json(series.to_json()).time_index
        if path == "fill_missing":
            missing = days.delete([13, 14, 15, 16])
            series = TimeSeries.from_times_and_values(
                missing, np.arange(26.0), fill_missing_dates=True, freq="D"
            )
            return series.time_index
        if path == "generate_index":
            return tg.generate_index(start=days[0], length=30, freq="D")
        if path == "attribute_add_length":
            return tg.datetime_attribute_timeseries(
                days[:15], "day", add_length=15
            ).time_index
        if path == "naive_model":
            return NaiveSeasonal(K=1).fit(head).predict(15).time_index
        return LinearRegressionModel(lags=3).fit(head).predict(15).time_index

    DAILY_PATHS = {
        "times_and_values": slice(None),
        "time_col": slice(None),
        "append_values": slice(None),
        "append": slice(None),
        "shift": slice(15, None),
        "json": slice(None),
        "fill_missing": slice(None),
        "generate_index": slice(None),
        "attribute_add_length": slice(None),
        "naive_model": slice(15, None),
        "regression_model": slice(15, None),
    }

    @pytest.mark.parametrize("case", list(MIDNIGHT_CHANGES))
    @pytest.mark.parametrize("path", list(DAILY_PATHS))
    def test_daily_steps_where_clocks_change_at_midnight(self, case, path):
        days = local_days(case)
        result = self.run_daily_path(path, days)
        assert_same_times(result, days[self.DAILY_PATHS[path]])
