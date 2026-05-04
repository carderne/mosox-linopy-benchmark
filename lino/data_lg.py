from copy import deepcopy

from data_sm import DATA as SMALL_DATA, PARAM_DIMS  # noqa

# The large Atlantis instance is the small instance with expanded index sets.
DATA = deepcopy(SMALL_DATA)
DATA["sets"].update(  # ty:ignore[no-matching-overload]
    {
        "MODE_OF_OPERATION": ["1", "2", "3"],
        "EMISSION": ["CO2", "NOx", "SO2", "CH4", "PM25"],
        "REGION": [
            "Atlantis_00A",
            "Atlantis_00B",
            "Atlantis_00C",
            "Atlantis_00D",
            "Atlantis_00E",
            "Atlantis_00F",
            "Atlantis_00G",
            "Atlantis_00H",
        ],
        "STORAGE": ["DAM", "BATTERY", "HYDROGEN"],
    }
)
