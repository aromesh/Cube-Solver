import csv

def writefile(filename, color_dict):

    with open(filename, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        for key, val in color_dict.items():
             csvwriter.writerow([key, val])


def readfile(filename, color_dict):

    dictionary = {}

    with open(filename, 'r') as f:
        data = list(csv.reader(f))
        for row in data:
            key = row[0]
            value = [int(s) for s in row[1][1:-1].split(',')]
            dictionary[key] = value

    return dictionary