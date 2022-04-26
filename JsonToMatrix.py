import json

def ConvertJsonToArray(filename):
    temp = []

    def JsonToListOfLists(arr):
        if type(arr) == list and (type(arr[0]) == int or type(arr[0]) == float):
            temp.append(arr)
        elif type(arr) == list:
            for i in arr:
                JsonToListOfLists(i)
        elif type(arr) == int or type(arr) == float:
            temp.append([arr])
        elif type(arr) == dict:
            for i in arr.values():
                JsonToListOfLists(i)

    vector = []
    matrix = []

    def CreateMatrix(arr, len, pos):
        if pos == len:
            matrix.append(vector[:])
        else:
            for i in arr[pos]:
                vector.append(i)
                CreateMatrix(arr, len, pos + 1)
                vector.pop()

    file = open(filename)
    data = json.loads(file.read())

    JsonToListOfLists(data)
    CreateMatrix(temp, len(temp), 0)

    return matrix

x = ConvertJsonToArray("input.txt")
for i in x:
    print(*i)