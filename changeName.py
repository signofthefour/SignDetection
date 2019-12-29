# Function to rename multiple file
# in a directory by Python3
import os


def main():
    i = 0;
    for filename in os.listdir("/home/nguyendat/Pictures/"):
        dst = "trafficLight" + str(i) + ".png"
        src = "/home/nguyendat/Pictures/" + filename
        dst = "/home/nguyendat/Pictures/" + dst
        os.rename(src, dst)
        i += 1


if __name__ == "__main__":
    main()