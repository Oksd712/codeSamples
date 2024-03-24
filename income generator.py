def calculate_population_distribution(total_population, avg_income, median_income, subdivision_data):
    # Define the income proportions
    income_proportions = {
        'Low-Income': 0.304,
        'Lower-Middle-Income': 0.225,
        'Upper-Middle-Income': 0.207,
        'High-Income': 0.141
    }

    result = {}

    # Iterate over each subdivision
    for subdivision in subdivision_data:
        sub_population = subdivision['Population']
        sub_avg_income = subdivision['Per capita income']
        sub_median_income = subdivision['median income']

        # Initialize data structure for subdivision
        result[subdivision['subdivision_name']] = {}

        # Calculate population distribution in each income category
        for category, proportion in income_proportions.items():
            # Adjust the proportion based on the subdivision's average and median income
            adjusted_proportion = proportion * (sub_avg_income / avg_income) * (sub_median_income / median_income)
            # Multiply the adjusted proportion by the subdivision's population
            result[subdivision['subdivision_name']][category] = int(adjusted_proportion * sub_population)

    return result

# Additional subdivision data
subdivision_data_additional = [
    {'subdivision_name': 'Civil Lines', 'Population': 266385, 'Per capita income': 85000, 'median income': 75000},
    {'subdivision_name': 'Gandhi Nagar', 'Population': 325650, 'Per capita income': 90000, 'median income': 84000},
    {'subdivision_name': 'Chanakyapuri', 'Population': 103675, 'Per capita income': 120000, 'median income': 120000},
    {'subdivision_name': 'Alipur', 'Population': 132567, 'Per capita income': 80000, 'median income': 66000},
    {'subdivision_name': 'Karawal Nagar', 'Population': 225789, 'Per capita income': 75000, 'median income': 60000},
    {'subdivision_name': 'Kanjhawala', 'Population': 153891, 'Per capita income': 90000, 'median income': 54000},
    {'subdivision_name': 'Seemapuri', 'Population': 225789, 'Per capita income': 75000, 'median income': 48000},
    {'subdivision_name': 'Hauz Khas', 'Population': 123567, 'Per capita income': 120000, 'median income': 105000},
    {'subdivision_name': 'Defence Colony', 'Population': 112356, 'Per capita income': 85000, 'median income': 150000},
    {'subdivision_name': 'Dwarka', 'Population': 266385, 'Per capita income': 85000, 'median income': 81000},
    {'subdivision_name': 'Patel Nagar', 'Population': 325650, 'Per capita income': 90000, 'median income': 78000},
    {'subdivision_name': 'Karol Bagh', 'Population': 295841, 'Per capita income': 110000, 'median income': 90000},
    {'subdivision_name': 'Mayur Vihar', 'Population': 385374, 'Per capita income': 125000, 'median income': 78000},
    {'subdivision_name': 'Delhi Cantonment', 'Population': 110464, 'Per capita income': 160000, 'median income': 105000},
    {'subdivision_name': 'Model Town', 'Population': 274734, 'Per capita income': 130000, 'median income': 96000},
    {'subdivision_name': 'Seelampur', 'Population': 554760, 'Per capita income': 75000, 'median income': 48000},
    {'subdivision_name': 'Rohini', 'Population': 811389, 'Per capita income': 95000, 'median income': 60000},
    {'subdivision_name': 'Shahdara', 'Population': 389970, 'Per capita income': 80000, 'median income': 54000},
    {'subdivision_name': 'Mehrauli', 'Population': 197414, 'Per capita income': 105000, 'median income': 105000},
    {'subdivision_name': 'Kalkaji', 'Population': 179679, 'Per capita income': 145000, 'median income': 114000},
    {'subdivision_name': 'Kapashera', 'Population': 167269, 'Per capita income': 90000, 'median income': 75000},
    {'subdivision_name': 'Punjabi Bagh', 'Population': 256374, 'Per capita income': 155000, 'median income': 99000},
    {'subdivision_name': 'Kotwali', 'Population': 234567, 'Per capita income': 135000, 'median income': 60000},
    {'subdivision_name': 'Preet Vihar', 'Population': 345678, 'Per capita income': 140000, 'median income': 84000},
    {'subdivision_name': 'Vasant Vihar', 'Population': 156789, 'Per capita income': 175000, 'median income': 120000},
    {'subdivision_name': 'Narela', 'Population': 765432, 'Per capita income': 90000, 'median income': 54000},
    {'subdivision_name': 'Yamuna Vihar', 'Population': 432109, 'Per capita income': 70000, 'median income': 66000},
    {'subdivision_name': 'Saraswati Vihar', 'Population': 210987, 'Per capita income': 95000, 'median income': 72000},
    {'subdivision_name': 'Vivek Vihar', 'Population': 321098, 'Per capita income': 115000, 'median income': 78000},
    {'subdivision_name': 'Saket', 'Population': 165432, 'Per capita income': 165000, 'median income': 135000},
    {'subdivision_name': 'Sarita Vihar', 'Population': 254321, 'Per capita income': 90000, 'median income': 90000},
    {'subdivision_name': 'Najafgarh', 'Population': 876543, 'Per capita income': 75000, 'median income': 75000},
    {'subdivision_name': 'Rajouri Garden', 'Population': 345678, 'Per capita income': 190000, 'median income': 96000},
    # Add more subdivisions as needed
]

total_population = 19800000
avg_income = 444000
median_income = 360000

# Merge existing and additional subdivision data
subdivision_data = subdivision_data_additional

result = calculate_population_distribution(total_population, avg_income, median_income, subdivision_data)
print(result)
