# tests/test_memory.py
from chae.memory import EpisodicMemory, SemanticMemory, WorkingMemory
from chae.perception import SentenceTransformerPerception


def test_working_memory_bounded():
    wm = WorkingMemory(max_size=3)
    for i in range(5):
        wm.push(i)
    assert len(wm) == 3 and wm.recent(10) == [2, 3, 4]


def test_episodic_recall_identical_score():
    p = SentenceTransformerPerception(fallback=True)
    em = EpisodicMemory(p)
    em.store("rain causes wet ground")
    assert em.recall(p.embed("rain causes wet ground"), top_k=1)[0]["score"] > 0.9


def test_semantic_query_finds_entity():
    sm = SemanticMemory()
    sm.add_entities(["rain", "wet_ground"], [("rain", "wet_ground")])
    res = sm.query("why does rain happen?")
    assert "rain" in res["entities"]