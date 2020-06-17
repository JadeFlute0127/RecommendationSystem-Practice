#加载数据
import csv

path = 'C:\\Users\\张典典\\Desktop\\ml-1m\\movies.csv'

f = open('C:\\Users\\张典典\\Desktop\\ml-1m\\rmovies_new.csv', 'w', encoding='utf-8',newline="")
csv_writer = csv.writer(f)
# csv_writer.writerow(["userId", "movieId", "rating", "timestamp"])
csv_writer.writerow(["movieId", "title", "genres"])

def loadCSV(filename):
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)

        for line in csvReader:
            tmp = ""
            for ss in line:
                tmp += ss

            movieId, title, genres = tmp.strip().split("::")
            csv_writer.writerow([movieId, title, genres])

    return dataSet

if __name__ == '__main__':

    loadCSV(path)
    f.close()
