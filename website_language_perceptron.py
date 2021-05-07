import numpy
import requests
import string
import math
import functools
from bs4 import BeautifulSoup
from requests.exceptions import MissingSchema, InvalidSchema


class Perceptron:
    def __init__(self, data_service, file_path, learning_value):
        self.data_service = data_service
        self.weights_vector = numpy.ones(len(string.ascii_lowercase) + 1)
        self.frequency_matrix = data_service.count_letters_frequencies_from_file(file_path)
        self.LEARNING_CONSTANT = learning_value

    def learn(self, number_of_iteration):
        print("Learning...\n")

        for iteration in range(number_of_iteration):
            for vector in self.frequency_matrix:
                if vector[0] == "pl":
                    expected = 1
                else:
                    expected = -1

                perceptron_output = self.test(numpy.array(vector[1:] + [-1]))

                self.weights_vector = self.weights_vector + (expected - perceptron_output) * self.LEARNING_CONSTANT * numpy.array(vector[1:] + [-1])
                self.weights_vector = (self.weights_vector / math.sqrt(sum(self.weights_vector ** 2))) * 10

    def test(self, vector):
        scalar = sum(self.weights_vector * vector)

        if scalar > 1:
            return 1
        elif scalar < (-1):
            return -1
        return scalar

    def check_website_language(self, url, get_content):
        perceptron_output = self.test(self.data_service.count_letters_frequency(get_content(url)) + [-1])
        if -0.1 < perceptron_output < 0.1:
            result = "HARD TO CLASSIFY"
        elif perceptron_output >= 0.1:
            result = "POLISH"
        else:
            result = "NOT POLISH"

        print(f"URL: {url}, language: {result}\n")


class DataService:
    letters_dictionary = {letter: 0 for letter in string.ascii_lowercase}

    def count_letters_frequencies_from_file(self, file_path):
        print("Frequency counting...")
        print("(This may take a while)\n")
        frequency_matrix = []

        f = open(file_path, "r")
        for line in f.readlines():
            if line.startswith("pl"):
                frequency_matrix.append(["pl"] + self.count_letters_frequency(get_content_from_url(line.split(" ")[1])))
            else:
                frequency_matrix.append(["not_pl"] + self.count_letters_frequency(get_content_from_url(line.split(" ")[1])))

        return frequency_matrix

    @functools.lru_cache()
    def count_letters_frequency(self, text):
        letters_sum = 0

        for key in self.letters_dictionary:
            counter = text.lower().count(key)
            letters_sum += counter
            self.letters_dictionary[key] = counter

        for key in self.letters_dictionary:
            self.letters_dictionary[key] = round(self.letters_dictionary[key] / letters_sum, 4)

        return list(self.letters_dictionary.values())


def get_content_from_url(url):
    html_content = requests.request("GET", url.removesuffix("\n")).text
    return ' '.join(BeautifulSoup(html_content, "html.parser").stripped_strings)


def main():
    perceptron = Perceptron(DataService(), r".\training_pages.txt", 0.1)
    perceptron.learn(1000)

    option = ""
    while option.__ne__("END"):
        option = input("Enter url address or 'END': ")
        try:
            perceptron.check_website_language(option, get_content_from_url)
        except (MissingSchema, InvalidSchema):
            if option.__ne__("END"):
                print("Invalid url address!")


main()
