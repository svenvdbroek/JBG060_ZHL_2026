import numpy as np
import matplotlib.pyplot as plt
import os
from processing_data.loading_impact_data import load_worldpop_area
from processing_data.loading_impact_data import load_admin_boundaries

admin1, admin2 = load_admin_boundaries()

bounds = admin1.total_bounds
bbox = {
    "lon_min": bounds[0],
    "lat_min": bounds[1],
    "lon_max": bounds[2],
    "lat_max": bounds[3],
}

sudan_pop = load_worldpop_area(bbox)

years = list(sudan_pop.keys())
population = list(sudan_pop.values())

plt.figure(figsize=(9,5))
plt.plot(years, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population of South Sudan')

os.makedirs('images', exist_ok=True)
plt.savefig('images/sudan_population.png', dpi=150, bbox_inches = 'tight')