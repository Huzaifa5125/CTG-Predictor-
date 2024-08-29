# display.py
from ydata_profiling import ProfileReport
import pandas as pd

# Load your data
df = pd.read_csv("fetal_health.csv")

# Generate the report
profile = ProfileReport(df, title="Fetal Health Classification")

# Save the report as an HTML file
profile.to_file("templates/DataANA.html")
