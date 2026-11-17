"""
Common-Sense Knowledge Base in Hyperon AtomSpace format.
Contains taxonomic hierarchies, habitats, biological capabilities, and diet relations.
Provides an intuitive, universally verifiable benchmark domain.
"""

from hyperon_miner.atomspace.types import (
    AtomSpace, ConceptNode, PredicateNode, ListLink, InheritanceLink, EvaluationLink
)


def load_commonsense_atomspace() -> AtomSpace:
    """Builds and populates a rich common-sense AtomSpace."""
    space = AtomSpace()

    # Entities
    mammals = ["human", "dog", "dolphin", "whale", "bat"]
    birds = ["sparrow", "eagle", "penguin"]
    reptiles = ["snake", "crocodile"]
    fish = ["salmon", "shark"]

    # Taxa concepts
    c_mammal = ConceptNode("mammal")
    c_bird = ConceptNode("bird")
    c_reptile = ConceptNode("reptile")
    c_fish = ConceptNode("fish")
    c_animal = ConceptNode("animal")

    # Predicates
    p_lives_in = PredicateNode("lives_in")
    p_breathes = PredicateNode("breathes")
    p_can_do = PredicateNode("can_do")
    p_eats = PredicateNode("eats")

    # Environmental / Attribute concepts
    c_air = ConceptNode("air")
    c_water = ConceptNode("water")
    c_ocean = ConceptNode("ocean")
    c_land = ConceptNode("land")
    c_mountain = ConceptNode("mountain")
    c_sky = ConceptNode("sky")

    c_fly = ConceptNode("fly")
    c_swim = ConceptNode("swim")
    c_walk = ConceptNode("walk")

    c_plants = ConceptNode("plants")
    c_meat = ConceptNode("meat")
    c_fish_food = ConceptNode("fish")
    c_plankton = ConceptNode("plankton")
    c_seeds = ConceptNode("seeds")

    # 1. Taxonomies (InheritanceLinks)
    for m in mammals:
        space.add(InheritanceLink((ConceptNode(m), c_mammal)))
        space.add(InheritanceLink((c_mammal, c_animal)))

    for b in birds:
        space.add(InheritanceLink((ConceptNode(b), c_bird)))
        space.add(InheritanceLink((c_bird, c_animal)))

    for r in reptiles:
        space.add(InheritanceLink((ConceptNode(r), c_reptile)))
        space.add(InheritanceLink((c_reptile, c_animal)))

    for f in fish:
        space.add(InheritanceLink((ConceptNode(f), c_fish)))
        space.add(InheritanceLink((c_fish, c_animal)))

    # 2. Respiration (EvaluationLink breathes ...)
    # All mammals, birds, reptiles breathe air
    for animal in mammals + birds + reptiles:
        space.add(EvaluationLink((p_breathes, ListLink((ConceptNode(animal), c_air)))))
    # Fish breathe water (dissolved oxygen)
    for animal in fish:
        space.add(EvaluationLink((p_breathes, ListLink((ConceptNode(animal), c_water)))))

    # 3. Habitats (EvaluationLink lives_in ...)
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("human"), c_land)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("dog"), c_land)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("bat"), c_land)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("dolphin"), c_ocean)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("whale"), c_ocean)))))

    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("sparrow"), c_land)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("eagle"), c_mountain)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("penguin"), c_land)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("penguin"), c_ocean)))))

    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("snake"), c_land)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("crocodile"), c_water)))))

    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("salmon"), c_water)))))
    space.add(EvaluationLink((p_lives_in, ListLink((ConceptNode("shark"), c_ocean)))))

    # 4. Locomotion / Capabilities (EvaluationLink can_do ...)
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("bat"), c_fly)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("eagle"), c_fly)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("sparrow"), c_fly)))))

    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("dolphin"), c_swim)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("whale"), c_swim)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("penguin"), c_swim)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("shark"), c_swim)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("salmon"), c_swim)))))

    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("human"), c_walk)))))
    space.add(EvaluationLink((p_can_do, ListLink((ConceptNode("dog"), c_walk)))))

    # 5. Diet (EvaluationLink eats ...)
    space.add(EvaluationLink((p_eats, ListLink((ConceptNode("dolphin"), c_fish_food)))))
    space.add(EvaluationLink((p_eats, ListLink((ConceptNode("whale"), c_plankton)))))
    space.add(EvaluationLink((p_eats, ListLink((ConceptNode("dog"), c_meat)))))
    space.add(EvaluationLink((p_eats, ListLink((ConceptNode("sparrow"), c_seeds)))))
    space.add(EvaluationLink((p_eats, ListLink((ConceptNode("eagle"), c_meat)))))
    space.add(EvaluationLink((p_eats, ListLink((ConceptNode("penguin"), c_fish_food)))))

    return space
