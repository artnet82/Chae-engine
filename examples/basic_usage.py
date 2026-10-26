"""Basic CHAE usage."""
from chae import CHAE, CHAEConfig
from chae.visualization import serve_in_background


def mock_web_search(query: str = "") -> dict:
    return {"results": [f"mock result for: {query}"]}


def main() -> None:
    engine = CHAE(CHAEConfig(perception_backend="sentence-transformers"))
    engine.tools.register("web_search", mock_web_search)
    r = engine.think("Why does rain make the ground wet?")
    print("Confidence:", r["causal_confidence"])
    print("Explanation:", r["explanation"])
    for s in r["plan"]["steps"]:
        print("  -", s)
    print("\nIntervene rain=False:", engine.intervene("rain", False))
    print("\nCounterfactual fire vs no_fire:", engine.counterfactual("fire", "no_fire"))
    print("\n[viz] http://127.0.0.1:8050")
    serve_in_background(engine.causal)


if __name__ == "__main__":
    main()