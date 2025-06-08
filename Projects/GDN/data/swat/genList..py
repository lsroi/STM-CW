import pandas as pd
def main():

    train = pd.read_csv('train.csv', index_col=0)

    f = open('list.txt', 'w')
    for col in train.columns:
        if col != 'attack' and col != 'timestamp':
            f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()