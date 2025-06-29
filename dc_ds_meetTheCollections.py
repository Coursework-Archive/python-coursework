# from collections import Counter
# nyc_eatery_count_by_types = Counter(nyc_eatery_types)
# print(nyc_eatery_count_by_types)
# #counter[key] = value
#
# #counter to find the most common
# #.most_common() method returns the counter values in descending order
# print(nyc_eatery_count_by_types.most_common(3))
# #[('Mobile Food Truch', 114), ('Food Cart', 74), ('Snack Bar', 24)]
#
# # Import the Counter object
# from collections import Counter
#
# # Create a Counter of the stations list: station_count
# station_count = Counter(stations)
#
# # Find the 5 most common elements
# print(station_count.most_common(5))
#
# #Dictionaries of unknown structure
#
# for park_id, name in nyc_eateries_parks:
#     if park_id not in eateries_by_park:
#         eateries_by park[park_id] = []
#     eateries_by_park[park_id].append(name)
# print(eateries_by_park['M010'])
#
# #Collections make the above algorithm much simpler
# from collections import defaultdict
# eateries_by_park = defaultdict(list)
# for park_id, name in nyc_eateries_parks:
#     eateries_by_park[park_id].append(name)
#
# print(eateries_by_park['M010'])
#
# #Here is another example
# from collections import defaultdict
# eatery_contact_types = defaultdict(int)
# for eatery in nyc_eateries:
#     if eatery.get('phone'):
#         eatery_contact_types['phones'] += 1
#     if eatery.get('website'):
#         eatery_contact_types['websites'] += 1
# print(eatery_contact_types)
#
# # Import the Counter object
# from collections import Counter
#
# # Print the first ten items from the stations list
# print(stations[:10])
#
# # Create a Counter of the stations list: station_count
# station_count = Counter(stations)
#
# # Print the station_count
# print(station_count)
#
# # Create an empty dictionary: ridership
# ridership = {}
#
# # Iterate over the entries
# for date, stop, riders in entries:
#     # Check to see if date is already in the ridership dictionary
#     if date not in ridership:
#         # Create an empty list for any missing date
#         ridership[date] = []
#     # Append the stop and riders as a tuple to the date keys list
#     ridership[date].append((stop, riders))
#
# # Print the ridership for '03/09/2016'
# print(ridership['03/09/2016'])
#
# #defaultdict allows you to define what each uninitialized key will contain.
# #When establishing a defaultdic you pass it the type you want it to be, such as
# #a list, tuple, set, int, string, dictionary
# # Import defaultdict
# from collections import defaultdict
#
# # Create a defaultdict with a default type of list: ridership
# ridership = defaultdict(list)
#
# # Iterate over the entries
# for date, stop, riders in entries:
#     # Use the stop as the key of ridership and append the riders to its value
#     ridership[stop].append(riders)
#
# # Print the first 10 items of the ridership dictionary
# print(list(ridership.items())[:10])
#
# #Python version < 3.6 NOT ordered
# #Python version > 3.6 ordered
#
# from collections import OrderedDict
# nyc_eatery_permits = OrderedDict()
# for eatery in nyc_eateries:
#     nyc_eatery_permits[eatery['end_date']] = eatery
# #.popitem() method returns items in reverse insertion order
# print(nyc_eatery_permits.popitem())
# # a second popitem returns the next last expiration
# print(nyc_eatery_permits.popitem())
#
# #you can use the last=False keyword argument to return the items in insertion order
# print(nyc_eatery_permits.popitem(last=False))
#
# # Import OrderedDict from collections
# from collections import OrderedDict
#
# # Create an OrderedDict called: ridership_date
# ridership_date = OrderedDict()
#
# # Iterate over the entries
# for date, riders in entries:
#     # If a key does not exist in ridership_date, set it to 0
#     if date not in ridership_date:
#         ridership_date[date] = 0
#
#     # Add riders to the date key in ridership_date
#     ridership_date[date] += riders
#
# # Print the first 31 records
# print(list(ridership_date.items())[:31])
#
# # Print the first key in ridership_date
# print(list(ridership_date.keys())[0])
#
# # Pop the first item from ridership_date and print it
# print(ridership_date.popitem(last=False))
#
# # Print the last key in ridership_date
# print(list(ridership_date.keys())[-1])
#
# # Pop the last item from ridership_date and print it
# print(ridership_date.popitem())
#
# #namedtuple - a tuple where each position (column) has a name
# #Ensure each one has the same properties
# #Alternative to a pandas DataFrame row
#
# from collections import namedtuple
# Eatery = namedtuple('Eatery', ['name', 'location', 'park_id', ... 'type_name'])
# eateries = []
# for eatery in nyc_eateries:
#     details = Eatery(eatery['name'], eatery['location'], eatery['park_id'], eatery['type_name'])
#     eateries.append(details)
#
# #print the first eatery in the list
# print(eateries[0])
# for eatery in eateries[:3]:
#     print(eatery.name)
#     print(eatery.park_id)
#     print(eatery.location)
#
# # Import namedtuple from collections
# from collections import namedtuple
#
# # Create the namedtuple: DateDetails
# DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])
# print(DateDetails)
#
# # Create the empty list: labeled_entries
# labeled_entries = []
#
# # Iterate over the entries list
# for date, stop, riders in entries:
#     # Append a new DateDetails namedtuple instance for each entry to labeled_entries
#     labeled_entries.append(DateDetails(date, stop, riders))
#
# # Print the first 5 items in labeled_entries
# print(labeled_entries[:5])
#
# # Iterate over the first twenty items in labeled_entries
# for item in labeled_entries[:20]:
#     # Print each item's stop
#     print(item.stop)
#
#     # Print each item's date
#     print(item.date)
#
#     # Print each item's riders
#     print(item.riders)
#
# #The datetime module is part of the Python standard library
# #Use the datetime type from inside the datetime module
# #.strptime() method converts from a string to a datetime object
#
# from datetime import datetime
# print(parking_violations_date)
# #Paarsing strings into datetimes
# date_dt = datetime.strptime(parking_violations_date, '%m%d%Y')
# print(date_dt)
#
# #.strftime() method uses a format string to convert a datetime object to a string
# date_dt.strftime('%m/%d/%Y')
#
# # Import the datetime object from datetime
# from datetime import datetime
#
# # Iterate over the dates_list
# for date_str in dates_list:
#     # Convert each date to a datetime object: date_dt
#     date_dt = datetime.strptime(date_str, '%m/%d/%Y')
#
#     # Print each date_dt
#     print(date_dt)
#
# # Loop over the first 10 items of the datetimes_list
# for item in datetimes_list[:10]:
#     # Print out the record as a string in the format of 'MM/DD/YYYY'
#     print(item.strftime('%m/%d/%Y'))
#
#     # Print out the record as an ISO standard string
#     print(item.isoformat())
#
# #Datetime Components
# #day, month, year, hour, minute, second
# #Great for grouping data

daily_violations =defaultdict(int)
for violation in parking_violations:
    violation_date = datetime.strptime(violation[4], '%m/%d/%Y')

    daily_violations[violation_date.day] += 1

    print(sorted(daily_violations.items()))

    #.now() method returns the current local datetime
    #utcnow() method returns the current UTC datetime

from datetime import datetime
local_dt = datetime.now()
print(local_dt)

utc_dt = datetime.utcnow()
print(utc_dt)

#Naive datetime objects have no timezone data
#Aware datetime objects have a timezone
#Timezone data is avaiable via the pytz module via the timezone object
#Aware objects have .astimezone() so you can get the time in another timezone

from pytz import timezone
record_dt = datetime.strptime('07/12/2016 04:39PM', ...: '%m/%d/%Y %H:%M%p')
ny_tz = timezone('US/Eastern')
la_tz = timezone('US/Pacific')
ny_dt = record_dt.replace(tzinfo=ny_tz)
la_dt = ny_dt.astimezone(la_tz)
print(ny_dt)
print(la_dt)
#
# # Create a defaultdict of an integer: monthly_total_rides
# monthly_total_rides = defaultdict(int)
#
# # Loop over the list daily_summaries
# for daily_summary in daily_summaries:
#     # Convert the service_date to a datetime object
#     service_datetime = datetime.strptime(daily_summary[0], '%m/%d/%Y')
#
#     # Add the total rides to the current amount for the month
#     monthly_total_rides[service_datetime.month] += int(daily_summary[4])
#
# # Print monthly_total_rides
# print(monthly_total_rides)

from datetime import datetime

print(datetime.now())

# Import datetime from the datetime module
from datetime import datetime

# Compute the local datetime: local_dt
local_dt = datetime.now()

# Print the local datetime
print(local_dt)

# Compute the UTC datetime: utc_dt
utc_dt = datetime.utcnow()

# Print the UTC datetime
print(utc_dt)

# Create a Timezone object for Chicago
chicago_usa_tz = timezone('US/Central')

# Create a Timezone object for New York
ny_usa_tz = timezone('US/Eastern')

# Iterate over the daily_summaries list
for orig_dt, ridership in daily_summaries:
    # Make the orig_dt timezone "aware" for Chicago
    chicago_dt = orig_dt.replace(tzinfo=chicago_usa_tz)

    # Convert chicago_dt to the New York Timezone
    ny_dt = chicago_dt.astimezone(ny_usa_tz)

    # Print the chicago_dt, ny_dt, and ridership
    print('Chicago: %s, NY: %s, Ridership: %s' % (chicago_dt, ny_dt, ridership))

#Adding and Subtracting time
#timedelta is used to represent an amount of change in time
#Used to add or subtract a set amount of time from a datetime object

from datetime import timedelta
flashback = timedelta(days=90)
print(record_dt)

print(record_dt - flashback)

print(record_dt + flashback)

time_diff = record_dt - record2_dt
type(time_diff)

print(time_diff)

# Import timedelta from the datetime module
from datetime import timedelta

# Build a timedelta of 30 days: glanceback
glanceback = timedelta(days=30)

# Iterate over the review_dates as date
for date in review_dates:
    # Calculate the date 30 days back: prior_period_dt
    prior_period_dt = date - glanceback

    # Print the review_date, day_type and total_ridership
    print('Date: %s, Type: %s, Total Ridership: %s' %
          (date,
           daily_summaries[date]['day_type'],
           daily_summaries[date]['total_ridership']))

    # Print the prior_period_dt, day_type and total_ridership
    print('Date: %s, Type: %s, Total Ridership: %s' %
          (prior_period_dt,
           daily_summaries[prior_period_dt]['day_type'],
           daily_summaries[prior_period_dt]['total_ridership']))

# Iterate over the date_ranges
for start_date, end_date in date_ranges:
    # Print the End and Start Date
    print(end_date, start_date)
    # Print the difference between each end and start date
    print(end_date - start_date)

#.parse() will attempt to convert a string to a pendulum datetime object without the need of the format string

import pendulum

occurred = violation[4] + ' ' + violation[5] + 'M'

occurred_dt = pendulum.parse(occurred, tz='US/Eastern')

print(occurred_dt)

#.in_timezone() method converts a pendulum time object to a desired timezone
#.now() method accepts a timezone you want to get the current time in
print(violation_dts)

for violation_dt in violation_dts:
    print(violation_dt.in_timezone('Asia/Tokyo'))

    print(pendulum.now('Asia/Tokyo'))

#.in_XXX() methods provide the difference in a chosen metric
#.in_words() provides the difference in a nice expressive form

diff = violation_dts[3] - violation_dts[2]

print(diff.in_words())

print(diff.in_days())

print(diff.in_hours())

# Import the pendulum module
import pendulum

# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now('Asia/Tokyo')

# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')

# Print the ISO 8601 string of la_dt
print(la_dt.to_iso8601_string())

# Iterate over date_ranges
for start_date, end_date in date_ranges:
    # Convert the start_date string to a pendulum date: start_dt
    start_dt = pendulum.parse(start_date, strict=False)

    # Convert the end_date string to a pendulum date: end_dt
    end_dt = pendulum.parse(end_date, strict=False)

    # Print the End and Start Date
    print(end_dt, start_dt)

    # Calculate the difference between end_dt and start_dt: diff_period
    diff_period = end_dt - start_dt

    # Print the difference in days
    print(diff_period.in_days())


import csv

csvfile = open('ART_GALLERY.csv', 'r')

for row in csv.reader(csvfile):
    print(row)

#Create and use a Counter with a slight twist

from collections import Counter

nyc_eatery_count_by_types = Counter(nyc_eatery_types)

#Use the date parts for Grouping like in Chapter 4

daily_violations = defaultdict(int)

for violation in paring_violations:
    violation_date = datetime.strptime(violation[4], '%m/%d/%Y')
    daily_violations[violation_date.date.day] += 1

from collections import defaultdict
eateries_by_park = default(list)

for park_id, name in nyc_eateries_parks:
    eateries_by_park[park_id].append(name)

#find the most common locations for crime each month
print(nyc_eatery_count_by_types.most_common(3))

# Import the csv module
import csv

# Create the file object: csvfile
csvfile = open('crime_sampler.csv', 'r')

# Create an empty list: crime_data
crime_data = []

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Append the date, type of crime, location description, and arrest
    crime_data.append((row[0], row[2], row[4], row[5]))

# Remove the first element from crime_data
crime_data.pop(0)

# Print the first 10 records
print(crime_data[:10])

# Import necessary modules
from collections import Counter
from datetime import datetime

# Create a Counter Object: crimes_by_month
crimes_by_month = Counter()

# Loop over the crime_data list
for row in crime_data:
    # Convert the first element of each item into a Python Datetime Object: date
    date = datetime.strptime(row[0], '%m/%d/%Y %I:%M:%S %p')
    print(date)
    # Increment the counter for the month of the row by one
    crimes_by_month[date.month] += 1

# Print the 3 most common months for crime
print(crimes_by_month.most_common(3))

# Import necessary modules
from collections import defaultdict
from datetime import datetime

# Create a dictionary that defaults to a list: locations_by_month
locations_by_month = defaultdict(list)

# Loop over the crime_data list
for row in crime_data:
    # Convert the first element to a date object
    date = datetime.strptime(row[0], '%m/%d/%Y %I:%M:%S %p')

    # If the year is 2016
    if date.year == 2016:
        # Set the dictionary key to the month and append the location (fifth element) to the values list
        locations_by_month[date.month].append(row[4])

# Print the dictionary
print(locations_by_month)

# Import Counter from collections
from collections import Counter

# Loop over the items from locations_by_month using tuple expansion of the month and locations
for month, locations in locations_by_month.items():
    # Make a Counter of the locations
    location_count = Counter(locations)
    # Print the month
    print(month)
    # Print the most common location
    print(location_count.most_common(5))

# Create the CSV file: csvfile
csvfile = open('crime_sampler.csv', 'r')

# Create a dictionary that defaults to a list: crimes_by_district
crimes_by_district = defaultdict(list)

# Loop over a DictReader of the CSV file
for row in csv.DictReader(csvfile):
    # Pop the district from each row: district
    district = row.pop('District')
    # Append the rest of the data to the list for proper district in crimes_by_district
    crimes_by_district[district].append(row)

#import csv

csvfile = open('ART_GALLERY.csv', 'r')

for row in csv.DictReader(csvfile):
    print(row)
#Pop out the key and store the remaining dict

galleries_10310 = art_galleries.pop('10310')

for zip_code, galleries in art_galeries.items():
    print(zip_code)
    print(galleries)

#Use sets for uniqueness
cookies_eaten_today = ['chocolate chip', 'peanut butte', 'chocolate chip', 'oatmeal cream' 'chocolate chip']

types_of_cookies_eaten = set(cookies_eaten_today)
print(types_of_cookies_eaten)

set(['chocolate chip', 'oatmeal cream', 'peanut butter'])

#difference() set mehod as at the end of Chapter 1
cookies_jason-ate.difference(cookies_hugo_ate)
set(['oatmeal cream', 'peanut butter'])

# Loop over the crimes_by_district using expansion as district and crimes
for district, crimes in crimes_by_district.items():
    # Print the district
    print(district)

    # Create an empty Counter object: year_count
    year_count = Counter()

    # Loop over the crimes:
    for crime in crimes:
        # If there was an arrest
        if crime['Arrest'] == 'true':
            # Convert the Date to a datetime and get the year
            year = datetime.strptime(crime['Date'], '%m/%d/%Y %I:%M:%S %p').year
            # Increment the Counter for the year
            year_count[year] += 1

    # Print the counter
    print(year_count)

# Create a unique list of crimes for the first block: n_state_st_crimes
n_state_st_crimes = set(crimes_by_block['001XX N STATE ST'])

# Print the list
print(n_state_st_crimes)

# Create a unique list of crimes for the second block: w_terminal_st_crimes
w_terminal_st_crimes = set(crimes_by_block['0000X W TERMINAL ST'])

# Print the list
print(w_terminal_st_crimes)

# Find the differences between the two blocks: crime_differences
crime_differences = n_state_st_crimes.difference(w_terminal_st_crimes)

# Print the differences
print(crime_differences)