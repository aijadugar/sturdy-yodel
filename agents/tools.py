from ddgs import DDGS

def hospital_search_tool(location: str):
    with DDGS() as ddgs:
        results = ddgs.text(
            f"nearest tuberculosis hospital near {location}",
            max_results=3
        )
        return [f"{r['title']} - {r['body']}" for r in results]


def lab_search_tool(location: str):
    with DDGS() as ddgs:
        results = ddgs.text(
            f"diagnostic lab for chest x-ray sputum test near {location}",
            max_results=3
        )
        return [f"{r['title']} - {r['body']}" for r in results]


def homecare_tool():
    return (
        "Home care guidance:\n"
        "- Take proper rest\n"
        "- Maintain nutrition and hydration\n"
        "- Wear a mask if coughing\n"
        "- Monitor symptoms for 7–10 days\n"
        "- Consult a doctor if symptoms worsen"
    )


def web_search_tool(query: str):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No relevant information found."

        return "\n\n".join(
            f"{r['title']}:\n{r['body']}" for r in results
        )