import numpy as np
import matplotlib.pyplot as plt

from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.inference import DecompositionalInference
import warnings

warnings.filterwarnings("ignore")

results = FuzzyVariable(
        universe_range=(0, 20),
        terms={
        'Mediocre': [(0, 1), (5, 0.5), (10, 0)],
        'Moyen': [(5, 0), (10, 1), (15, 0)],
        'Excellent': [(10, 0), (15, 0.5), (20, 1)]
    }
    )


methods = FuzzyVariable(
        universe_range=(0, 20),
        terms={
        'Mediocre': [(0, 1), (5, 0.5), (10, 0)],
        'Moyen': [(5, 0), (10, 1), (15, 0)],
        'Excellent': [(10, 0), (15, 0.5), (20, 1)]
    }
    )

presentation = FuzzyVariable(
        universe_range=(0, 20),
        terms={
        'Mediocre': [(0, 1), (5, 0.5), (10, 0)],
        'Moyen': [(5, 0), (10, 1), (15, 0)],
        'Excellent': [(10, 0), (15, 0.5), (20, 1)]
    }
    )

evaluation = FuzzyVariable(
        universe_range=(0, 20),
        terms={
        'Mediocre': [(0, 1), (5, 0.5), (10, 0)],
        'Mauvais': [(5, 0), (10, 0.5), (15, 0)],
        'Moyen': [(10, 0), (15, 1), (20, 0)],
        'Bon': [(15, 0), (17, 0.5), (20, 1)],
        'Excellent': [(17, 0), (20, 1)]
    }
    )

rules = [
    FuzzyRule(
        premise=[
            ("results", "Moyen"),
            ("AND", "methods", "Mediocre")
        ],
        consequence=[("evaluation", "Mauvais")]
    ),
    FuzzyRule(
        premise=[
            ("results", "Moyen"),
            ("AND", "methods", "Excellent")
        ],
        consequence=[("evaluation", "Bon")]
    ),
    FuzzyRule(
        premise=[
            ("results", "Mediocre"),
            ("AND", "methods", "Moyen")
        ],
        consequence=[("evaluation", "Mauvais")]
    ),
    FuzzyRule(
        premise=[
            ("results", "Excellent"),
            ("AND", "methods", "Excellent"),
            ("AND", "presentation", "Excellent")
        ],
        consequence=[("evaluation", "Excellent")]
    ),
    FuzzyRule(
        premise=[
            ("results", "Mediocre"),
            ("OR", "methods", "Moyen")
        ],
        consequence=[("evaluation", "Moyen")]
    ),
    FuzzyRule(
        premise=[
            ("results", "Moyen"),
            ("OR", "methods", "Mediocre")
        ],
        consequence=[("evaluation", "Mediocre")]
    )
]


def main():
    plt.figure(figsize=(10, 2.5))

    results.plot()
    methods.plot()
    presentation.plot()
    model = DecompositionalInference(
    and_operator="min",
    or_operator="max",
    implication_operator="Rc",
    composition_operator="max-min",
    production_link="max",
    defuzzification_operator="cog",)

    model({'results': results, 'methods': methods, 'presentation': presentation, 'evaluation':evaluation},
          rules,
          results=12, methods=15, presentation=18)
    
    print(model.infered_cf)
    model.plot({'results': results, 'methods': methods, 'presentation': presentation, 'evaluation':evaluation},
          rules,
          results=12, methods=15, presentation=18)
    plt.show()




if __name__ == "__main__":
    main()