"""Macro data for stress testing and the quarterly config.

    catalogue   the datasets, countries and concepts (packaged CSVs)
    sources     the IMF SDMX API, DataMapper, DBnomics, FRED, files, the
                package's own history
    store       the vintaged store on disk
    fetch       one refresh of the store
    quarter     a new quarter's config inputs from the store, into a draft
    system      the linked effects: how a scenario's main variables move the
                others (a recursive system estimated on outturns)
    var         an earlier macro model (a Bayesian VAR), kept for comparison

None of it is read by a run: the booked provision is priced from the run's
frozen config alone.
"""
from .catalogue import concepts, countries, datasets, match_candidates, series_id
from .store import MacroStore, annualise, to_concept_unit
from .fetch import import_file, refresh

__all__ = ["concepts", "countries", "datasets", "match_candidates", "series_id",
           "MacroStore", "annualise", "to_concept_unit", "import_file", "refresh"]
