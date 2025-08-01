import pickle
import os


class Person:
    __version__ = 1  # increment this whenever you change attributes

    def __init__(self, name, age, kids, occupation, salary, location):
        self.name = name
        self.age = age
        self.kids = kids
        self.occupation = occupation
        self.salary = salary
        self.location = location

    def __repr__(self):
        return (f"<Person name={self.name}, "
                f"age={self.age}, "
                f"kids={self.kids}, "
                f"occupation={self.occupation}, "
                f"salary={self.salary}, "
                f"location={self.location}>"
                )

    def __eq__(self, other):
        return (
                isinstance(other, Person)
                and self.name == other.name
                and self.age == other.age
                and self.kids == other.kids
                and self.occupation == other.occupation
                and self.salary == other.salary
                and self.location == other.location
        )


def serialize(obj):
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Serialized object: \n{pickled}\n')
    return pickled


def deserialize(obj):
    unpickled = pickle.loads(obj)
    print(f'Deserialized: \n{unpickled}\n')


def deserialize_occupation(obj):
    unpickled = pickle.loads(obj)
    print(f'Deserialized occupation: \n{unpickled.occupation}\n')


def obj_list_to_file(fn, obj_list):
    data = {
        "version": Person.__version__,
        "people": obj_list
    }
    with open(fn, 'wb') as pf:
        pickle.dump(data, pf, protocol=pickle.HIGHEST_PROTOCOL)  # ✅ Now saving the dictionary
    print(f"[SUCCESS] Written {len(obj_list)} object(s) to {fn}")


def obj_to_file(fn, obj):
    obj_list = file_to_obj_list(fn)

    if obj in obj_list:
        print(f"[SKIP] Person already exists in {fn}")
        return

    if not isinstance(obj_list, list):
        raise TypeError(f"Expected a list in {fn}, got {type(obj_list).__name__}")

    obj_list.append(obj)
    obj_list_to_file(fn, obj_list)
    print(f"[SUCCESS] Appended new person to {fn}")


def file_to_obj_list(fn):
    if not os.path.exists(fn):
        return []

    try:
        with open(fn, 'rb') as pf:
            data = pickle.load(pf)
    except Exception as e:
        print(f"[ERROR] Failed to load {fn}: {e}.")
        delete = input(f"[PROMPT] Delete {fn}? (y/n): ")
        if delete.lower() == 'y':
            try:
                os.remove(fn)
                print(f"[INFO] Deleted {fn}")
            except Exception as e2:
                print(f"[ERROR] Could not delete {fn}: {e2}")
        else:
            print(f"[SKIP] Keeping {fn}")
        return []

    # Handle versioned data
    if isinstance(data, dict) and "version" in data and "people" in data:
        if data["version"] != Person.__version__:
            print(f"[WARNING] Version mismatch in {fn} (expected v{Person.__version__}, found v{data['version']})")
            delete = input(f"[PROMPT] Delete outdated file {fn}? (y/n): ")
            if delete.lower() == 'y':
                try:
                    os.remove(fn)
                    print(f"[INFO] Deleted {fn}")
                except Exception as e:
                    print(f"[ERROR] Could not delete {fn}: {e}")
            else:
                print(f"[SKIP] Keeping {fn}")
            return []
        return data["people"]

    # Handle unexpected format
    print(f"[WARNING] Unexpected format in {fn}")
    delete = input(f"[PROMPT] Delete malformed file {fn}? (y/n): ")
    if delete.lower() == 'y':
        try:
            os.remove(fn)
            print(f"[INFO] Deleted {fn}")
        except Exception as e:
            print(f"[ERROR] Could not delete {fn}: {e}")
    else:
        print(f"[SKIP] Keeping {fn}")
    return []


def process_person(prsn, fn):
    if not hasattr(prsn, "name"):
        raise TypeError(f"Expected Person instance, got {type(prsn).__name__}")

    print(f"\n[PROCESSING] {prsn.name}")
    pickled = serialize(prsn)
    deserialize(pickled)
    deserialize_occupation(pickled)
    obj_to_file(fn, prsn)
    restored = file_to_obj_list(fn)
    return restored[-1] if restored else None


# === Usage ===
file_path = '../files_to_read/person.xyz'

person = Person(
    'John Smith',
    '45',
    ['Pete', 'Lilly', 'Kate'],
    'AWS',
    '$150,000.00/yr',
    'San Francisco, CA'
)
all_people = file_to_obj_list(file_path)
print(f"[INFO] Restored {len(all_people)} person(s):")
for p in all_people:
    print(" •", p)

person1 = Person(
    'Parker Mathew',
    '23',
    None,
    'Starbucks',
    '$49,240.00/yr',
    'San Francisco, CA'
)


process_person(person, file_path)
process_person(person1, file_path)
