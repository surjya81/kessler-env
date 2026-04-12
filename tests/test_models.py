"""
test_models.py — validates that KesslerAction, KesslerObservation, and
ThrusterBurn schemas accept valid data and reject invalid data correctly.
"""
import pytest
from models import KesslerAction, KesslerObservation, ThrusterBurn, SatelliteTelemetry, DebrisTelemetry


class TestThrusterBurn:
    def test_valid_burn(self):
        b = ThrusterBurn(satellite_id=0, delta_vx=0.5, delta_vy=-0.3)
        assert b.satellite_id == 0
        assert b.delta_vx == 0.5
        assert b.delta_vy == -0.3

    def test_missing_satellite_id_raises(self):
        with pytest.raises(Exception):
            ThrusterBurn(delta_vx=0.1, delta_vy=0.1)

    def test_zero_burn_valid(self):
        b = ThrusterBurn(satellite_id=1, delta_vx=0.0, delta_vy=0.0)
        assert b.delta_vx == 0.0


class TestKesslerAction:
    def test_empty_burns(self):
        a = KesslerAction(burns=[])
        assert a.burns == []

    def test_default_burns_is_empty(self):
        a = KesslerAction()
        assert a.burns == []

    def test_multiple_burns(self):
        a = KesslerAction(burns=[
            ThrusterBurn(satellite_id=0, delta_vx=0.1, delta_vy=0.0),
            ThrusterBurn(satellite_id=2, delta_vx=-0.5, delta_vy=0.5),
        ])
        assert len(a.burns) == 2


class TestKesslerObservation:
    def _make_obs(self, **kwargs):
        defaults = dict(
            satellites=[SatelliteTelemetry(id=0, x=50.0, y=0.0, vx=0.0, vy=4.5, fuel=100.0, status="active")],
            radar_debris=[],
        )
        defaults.update(kwargs)
        return KesslerObservation(**defaults)

    def test_minimal_valid_observation(self):
        obs = self._make_obs()
        assert obs.done is False
        assert obs.reward == 0.0

    def test_radar_range_defaults_to_zero(self):
        obs = self._make_obs()
        assert obs.radar_range == 0.0

    def test_radar_range_can_be_set(self):
        obs = self._make_obs(radar_range=42.0)
        assert obs.radar_range == 42.0

    def test_critical_alerts_defaults_empty(self):
        obs = self._make_obs()
        assert obs.critical_alerts == []

    def test_mission_objective_defaults_empty_string(self):
        obs = self._make_obs()
        assert obs.mission_objective == ""

    def test_target_radius_defaults_zero(self):
        obs = self._make_obs()
        assert obs.target_radius == 0.0

    def test_total_score_defaults_zero(self):
        obs = self._make_obs()
        assert obs.total_score == 0.0


class TestSatelliteTelemetry:
    def test_valid_satellite(self):
        s = SatelliteTelemetry(id=0, x=50.0, y=10.0, vx=1.0, vy=4.0, fuel=80.0, status="active")
        assert s.status == "active"

    def test_destroyed_status(self):
        s = SatelliteTelemetry(id=1, x=0.0, y=0.0, vx=0.0, vy=0.0, fuel=0.0, status="destroyed")
        assert s.status == "destroyed"


class TestDebrisTelemetry:
    def test_valid_debris(self):
        d = DebrisTelemetry(id=5, x=60.0, y=30.0, vx=-2.0, vy=3.5)
        assert d.id == 5
