import csv, os


def writeToCsv(content, heading = ['Medicine', 'Database Summary', 'General Summary']):

    '''
    content needs to be in the form of [{"Medicine": "name", "Database Summary": "stuff", "General Summary": "stuff"}, ...]
    '''

    i = 0
    while os.path.exists(f'Medicine Summary{i}.csv'):
        i += 1

    file = f'Medicine Summary{i}.csv'

    with open(file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = heading)
        writer.writeheader()
        writer.writerows(content)

    
