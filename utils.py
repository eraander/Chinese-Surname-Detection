def strip(filename):
    with open(filename, 'r') as f, open('data/dc_surname', 'a') as f2:
        line = f.readline()
        line = line.replace('"', ' ')
        line = line.replace(',', ' ')
        line = line.split()
        for token in line:
            print(token)
            f2.write(token + '\n')
        print("done!")



