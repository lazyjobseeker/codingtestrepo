import pytest

@pytest.fixture
def on_plate():
    return []

@pytest.fixture
def add_icecream(on_plate):
    on_plate.append("Icecream")
    raise ValueError

@pytest.fixture(autouse=True)
def add_waffle(on_plate, add_icecream):
    on_plate.append("Waffle")

def test_on_plate(on_plate):
    assert on_plate == ["Icecream", "Waffle"]