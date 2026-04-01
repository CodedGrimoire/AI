import osmnx as ox
import certifi

# Ensure SSL uses a valid CA bundle in restricted environments
ox.settings.requests_kwargs = {"verify": certifi.where()}

# Option A: by place name (preferred)
place = "Dhanmondi, Dhaka, Bangladesh"

try:
    G = ox.graph_from_place(place, network_type="drive")
except ValueError:
    # Fallback: small buffer around the area center if place query returns empty
    center = (23.746, 90.376)  # lat, lon for Dhanmondi Lake area
    G = ox.graph_from_point(center, dist=1500, network_type="drive")

G = ox.project_graph(G)  # planar coordinates for distance calculations

# Render to file (no GUI needed)
ox.plot_graph(
    G,
    show=False,
    close=True,
    save=True,
    filepath="map.png",
    dpi=150,
    edge_color="#4b8bbe",
    node_size=6,
)
