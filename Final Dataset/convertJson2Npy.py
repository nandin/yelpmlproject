import json
import numpy as np

numData = 2211
numFeatures = 11
dataset = np.zeros((numData, numFeatures))

with open('finalDataSet.json') as jsonData:
	"""
	All binary features are 1 for true and 0 for false
	feature 1: business ID
	feature 2: Postal Code - int
	feature 3: Review Count - int
	feature 4: Alcohol - binary
	feature 5: Wifi - int [0 for no, 1 for paid, 2 for free]
	feature 6: Good for Kids - binary
	feature 7: Delivery - binary
	feature 8: Reservation - binary
	feature 9: Take Out - binary
	feature 10: Price Range - int
	11th Column is the label column - stars 
	"""
	count = 0
	restID = 1
	for restaurant in jsonData.readlines():
		restaurant_dict = json.loads(restaurant)
		attributes = restaurant_dict["attributes"]
		restaurantData = np.zeros((1, numFeatures))

		# label
		restaurantData[0,10] = float(restaurant_dict["stars"])

		# restaurant id
		restaurantData[0,0] = restID

		# postal code
		restaurantData[0,1] = float(restaurant_dict["postal_code"])

		# review count
		restaurantData[0,2] = float(restaurant_dict["review_count"])

		# alcohol
		alcohol = attributes["Alcohol"]
		if alcohol == "u'none'" or alcohol == "'none'":
			alcoholBool = 0
		elif alcohol == "u'beer_and_wine'" or alcohol == "'beer_and_wine'" or alcohol == "u'full_bar'" or alcohol == "'full_bar'":
			alcoholBool = 1
		else:
			alcoholBool = 0
		restaurantData[0,3] = alcoholBool

		# wifi
		wifi = attributes["WiFi"]
		print(wifi)
		if wifi == "u'free'" or wifi == "'free'":
			wifiInt = 2
		elif wifi == "u'paid'" or wifi == "'paid'":
			wifiInt = 1
		else:
			wifiInt = 0
		print(wifiInt)
		restaurantData[0,4] = wifiInt

		# good for kids
		goodForKids = attributes["GoodForKids"]
		if goodForKids is None or goodForKids == 'False':
			restaurantData[0,5] = 0
		else:
			restaurantData[0,5] = 1

		# Delivery 
		delivery = attributes["RestaurantsDelivery"]
		if delivery is None or delivery == 'False':
			restaurantData[0,6] = 0
		else:
			restaurantData[0,6] = 1

		# Reservation
		reservation = attributes["RestaurantsReservations"]
		if reservation is None or reservation == 'False':
			restaurantData[0,7] = 0
		else:
			restaurantData[0,7] = 1

		# Take Out
		takeout = attributes["RestaurantsTakeOut"]
		if takeout is None or takeout == 'False':
			restaurantData[0,8] = 0
		else:
			restaurantData[0,8] = 1

		# Price Range
		priceRange = attributes["RestaurantsPriceRange2"]
		if priceRange is None or priceRange == '1':
			restaurantData[0,9] = 1
		elif priceRange == '2':
			restaurantData[0,9] = 2
		elif priceRange == '3':
			restaurantData[0,9] = 3
		else:
			restaurantData[0,9] = 4

		dataset[count, :] = restaurantData
		count += 1
		restID += 1
np.savetxt('dataset.csv', dataset, delimiter = ',')
np.save('dataset.npy', dataset)






