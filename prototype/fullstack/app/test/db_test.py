from pymongo import MongoClient
import datetime
import certifi

ca = certifi.where()

client = MongoClient('mongodb+srv://jadon:aiit@cluster0.13mh6.mongodb.net/myFirstDatabase?retryWrites=true&w=majority', tlsCAFile=ca)
db = client.gettingStarted
people = db.people
import datetime
personDocument = {
  "name": { "first": "Alan", "last": "Turing" },
  "birth": datetime.datetime(1912, 6, 23),
  "death": datetime.datetime(1954, 6, 7),
  "contribs": [ "Turing machine", "Turing test", "Turingery" ],
}
people.insert_one(personDocument)
print('here')