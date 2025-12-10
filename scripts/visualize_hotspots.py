"""
Visualization Script for CrimeHotspotSim
Author: Kayla Ramos

This script:

1. Loads model prediction outputs:
    - latest_scores_before.csv (baseline hotspot predictions)
    - latest_scores_after.csv (post-intervention predictions)
2. Generates interactive Folium maps (centered on Cleveland, OH)
3. Plots crime hotspot points on each map using CircleMarkers:
    * Green: Low 0-0.33
    * Orange: Med 0.33-0.66
    * Red: High 0.66-1
4. Adds a fixed HTML legend explaining the color coding for each risk score
5. Exports individual maps as HTML files
    - map_before.html (baseline)
    - map_after.html (post-intervention)
6. Combines both maps into a single HTML file
    - crime_hotspot_comparison.html
    - uses an <iframe> style layout so the two maps display simultaneously in a browser
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster

# -----------------------------
# Load predictions
# -----------------------------
before_csv = "predictions/latest_scores_before.csv"
after_csv = "predictions/latest_scores_after.csv"

df_before = pd.read_csv(before_csv)
df_after = pd.read_csv(after_csv)

# -----------------------------
# Center point for Cleveland
# -----------------------------
cleveland_center = [41.4993, -81.6944]  # lat, lon

# -----------------------------
# Function to get color based on risk score
# -----------------------------
def get_color(score):
    if score < 0.33:
        return "green"
    elif score < 0.66:
        return "orange"
    else:
        return "red"

# -----------------------------
# Function to add circle markers
# -----------------------------
def add_markers(fmap, df, score_col):
    cluster = MarkerCluster().add_to(fmap)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=get_color(row[score_col]),
            fill=True,
            fill_opacity=0.7
        ).add_to(cluster)

# -----------------------------
# Function to add legend
# -----------------------------
def add_legend(fmap, title="Risk Score"):
    legend_html = """
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 110px; 
     background-color: white; z-index:9999; font-size:14px; 
     border:2px solid grey; padding: 10px;
     ">
     <b>{}</b><br>
     <i style="background:green;color:green;">....</i> Low (0-0.33)<br>
     <i style="background:orange;color:orange;">....</i> Medium (0.33-0.66)<br>
     <i style="background:red;color:red;">....</i> High (0.66-1)<br>
     </div>
    """.format(title)
    fmap.get_root().html.add_child(folium.Element(legend_html))

# -----------------------------
# Create and save individual maps
# -----------------------------
m_before = folium.Map(location=cleveland_center, zoom_start=11, tiles="CartoDB positron")
add_markers(m_before, df_before, "score_before")
add_legend(m_before, "Baseline Risk")
m_before.save("predictions/map_before.html")

m_after = folium.Map(location=cleveland_center, zoom_start=11, tiles="CartoDB positron")
add_markers(m_after, df_after, "score_after")
add_legend(m_after, "Post-Intervention Risk")
m_after.save("predictions/map_after.html")

# -----------------------------
# Side-by-side HTML using <iframe>
# -----------------------------
html_output = """
<!DOCTYPE html>
<html>
  <head>
    <title>Crime Hotspot Comparison</title>
    <style>
      #map_container {
        display: flex;
        flex-direction: row;
        width: 100%;
        height: 100vh;
      }
      iframe {
        flex: 1;
        height: 100%;
        border: none;
      }
    </style>
  </head>
  <body>
    <div id="map_container">
      <iframe src="map_before.html"></iframe>
      <iframe src="map_after.html"></iframe>
    </div>
  </body>
</html>
"""

# -----------------------------
# Save final HTML
# -----------------------------
with open("predictions/crime_hotspot_comparison.html", "w") as f:
    f.write(html_output)

print("Saved: predictions/crime_hotspot_comparison.html")

