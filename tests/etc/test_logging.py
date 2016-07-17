import logging
import datetime


logging.basicConfig(filename='myapp.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logging.info('is when this event was logged.')

start = datetime.datetime.now()

for i in range(100000000):
    12940192809 ** 2

end = datetime.datetime.now()

delta = end - start

print(delta)