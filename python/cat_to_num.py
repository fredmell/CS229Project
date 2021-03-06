import numpy as np

cat_to_num_mapping = {'AcceptsInsurance': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0},
                      'AgesAllowed': {np.nan: np.nan, 'None': 0, '18plus': 1, '19plus': 2, '21plus': 3, 'allages': 4},
                      'Alcohol': {np.nan: np.nan, 'None': 0, 'none': 0, 'full_bar': 1, 'beer_and_wine': 1},
                      'Ambience': {np.nan: np.nan, 'None': 0},
                      'Ambience_casual': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'Ambience_classy': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_divey': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_hipster': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_intimate': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_romantic': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_touristy': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_trendy': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Ambience_upscale': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'BestNights': {np.nan: np.nan, 'None': 0},
                      'BestNights_friday': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'BestNights_monday': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'BestNights_saturday': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'BestNights_sunday': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'BestNights_thursday': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'BestNights_tuesday': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'BestNights_wednesday': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'BikeParking': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'BusinessAcceptsBitcoin': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'BusinessAcceptsCreditCards': {'True': 1, np.nan: np.nan, 'False': 0, 'None': 0},
                      'BusinessParking': {np.nan: np.nan, '{}': 0, 'None': 0},
                      'BusinessParking_garage': {'True': 1, 'False': 0, np.nan: np.nan},
                      'BusinessParking_lot': {'True': 1, 'False': 0, np.nan: np.nan},
                      'BusinessParking_street': {'True': 1, 'False': 0, np.nan: np.nan},
                      'BusinessParking_valet': {'True': 1, 'False': 0, np.nan: np.nan},
                      'BusinessParking_validated': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'ByAppointmentOnly': {'True': 1, np.nan: np.nan, 'False': 0, 'None': 0},
                      'BYOB': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'BYOBCorkage': {np.nan: np.nan, 'yes_free': 1, 'None': 0, 'yes_corkage': 1, 'no': 0},
                      'Caters': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'CoatCheck': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'Corkage': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'DietaryRestrictions': {np.nan: np.nan, 'None': 0},
                      'DietaryRestrictions_dairy-free': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'DietaryRestrictions_gluten-free': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'DietaryRestrictions_halal': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'DietaryRestrictions_kosher': {np.nan: np.nan, 'False': 0},
                      'DietaryRestrictions_soy-free': {np.nan: np.nan, 'False': 0},
                      'DietaryRestrictions_vegan': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'DietaryRestrictions_vegetarian': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'DogsAllowed': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'DriveThru': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'GoodForDancing': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'GoodForKids': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0},
                      'GoodForMeal': {np.nan: np.nan, '{}': 0, 'None': 0},
                      'GoodForMeal_breakfast': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'GoodForMeal_brunch': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'GoodForMeal_dessert': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'GoodForMeal_dinner': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'GoodForMeal_latenight': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'GoodForMeal_lunch': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'HairSpecializesIn': {np.nan: np.nan, 'None': 0},
                      'HairSpecializesIn_africanamerican': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'HairSpecializesIn_asian': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'HairSpecializesIn_coloring': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'HairSpecializesIn_curly': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'HairSpecializesIn_extensions': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'HairSpecializesIn_kids': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'HairSpecializesIn_perms': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'HairSpecializesIn_straightperms': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'HappyHour': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0},
                      'HasTV': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'Music': {np.nan: np.nan, '{}': 0, 'None': 0},
                      'Music_background_music': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Music_dj': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Music_jukebox': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Music_karaoke': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'Music_live': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'Music_no_music': {np.nan: np.nan, 'False': 0},
                      'Music_video': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'NoiseLevel': {np.nan: np.nan, 'None': 0, 'quiet': 1, 'average': 2, 'loud': 3, 'very_loud': 4},
                      'Open24Hours': {np.nan: np.nan, 'True': 1, 'False': 0},
                      'OutdoorSeating': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'RestaurantsAttire': {np.nan: np.nan, 'None': 0, 'casual': 1, 'dressy': 2, 'formal': 3},
                      'RestaurantsCounterService': {np.nan: np.nan, 'False': 0, 'True': 1},
                      'RestaurantsDelivery': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'RestaurantsGoodForGroups': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0},
                      'RestaurantsPriceRange2': {np.nan: np.nan, '1': 1, 'None': 0, '2': 2, '3': 3, '4': 4},
                      'RestaurantsReservations': {np.nan: np.nan, 'False': 0, 'None': 0, 'True': 1},
                      'RestaurantsTableService': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0, 'none': 0},
                      'RestaurantsTakeOut': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0},
                      'Smoking': {np.nan: np.nan, 'None': 0, 'outdoor': 1, 'yes': 1, 'no': 0},
                      'WheelchairAccessible': {np.nan: np.nan, 'True': 1, 'False': 0, 'None': 0},
                      'WiFi': {np.nan: np.nan, 'paid': 0, 'None': 0, 'free': 1, 'no': 0}
                      }
