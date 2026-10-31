import numpy as np

from pulsescope import config as C
from pulsescope import simulate


def test_users_deterministic_and_personas_present():
    a = simulate.make_users(2000, np.random.default_rng(1))
    b = simulate.make_users(2000, np.random.default_rng(1))
    assert a.equals(b)
    assert set(a.persona_idx.unique()) == set(range(len(simulate.PERSONAS)))


def test_incident_only_on_android_520():
    users = simulate.make_users(5000, np.random.default_rng(2))
    ev = simulate.simulate_day(C.INCIDENT_START + 1, users, np.random.default_rng(3))
    assert set(ev.loc[ev.app_version == "5.2.0", "platform"]) == {"android"}
    ev_before = simulate.simulate_day(C.INCIDENT_START - 1, users, np.random.default_rng(3))
    assert "5.2.0" not in set(ev_before.app_version)


def test_dirty_rows_injected():
    users = simulate.make_users(3000, np.random.default_rng(4))
    ev = simulate.simulate_day(10, users, np.random.default_rng(5))
    assert ev.event_id.duplicated().any()
    assert ev.user_id.isna().any()
    assert (ev.user_id.dropna() < 0).any()
