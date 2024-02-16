import os
import urllib.request
import csv
import json
import urllib.request
from sklearn.metrics import f1_score
import cv2 as cv
import numpy as np


def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image

def find_keypoints_and_descriptors_sift(image_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def find_keypoints_and_descriptors_orb(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def compare_sift(image1_path, image2_path):
    kp1, desc1 = find_keypoints_and_descriptors_sift(image1_path)
    kp2, desc2 = find_keypoints_and_descriptors_sift(image2_path)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches)

def compare_orb(url1, url2):
    image1 = load_image_from_url(url1)
    image2 = load_image_from_url(url2)

    kp1, desc1 = find_keypoints_and_descriptors_orb(image1)
    kp2, desc2 = find_keypoints_and_descriptors_orb(image2)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches)


def compare_images_txt(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    output_file_path = 'results\\output_test_results_orb.txt'
    total_pairs = len(data['data']['results'])
    counter = 0
    with open(output_file_path, 'w') as output_file:
        for result in data['data']['results']:
            image_url1 = result['representativeData']['image1']['imageUrl']
            image_url2 = result['representativeData']['image2']['imageUrl']

            dist = compare_orb(image_url1, image_url2)
            answer = result['answers'][0]['answer'][0]['id']
            output_file.write(f"Distance between images: {dist}  Valid answer: {answer}\n")

            counter += 1
            print(f"{counter}/{total_pairs}")

def compare_images_csv(json_file_path, images_folder_path):
    csv_file_path = 'results\\output_results2.csv'
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    total_pairs = len(data['data']['results'])
    counter = 0
    with open(csv_file_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['taskId', 'answer'])
        for result in data['data']['results']:
            task_id = result['taskId']
            image1_path = os.path.join(images_folder_path, f"{task_id}_1.jpg")
            image2_path = os.path.join(images_folder_path, f"{task_id}_2.jpg")

            dist_sift = compare_sift(image1_path, image2_path)
            answer = 1 if dist_sift >= 90 else 0
            csvwriter.writerow([task_id, answer])

            counter += 1
            print(f"{counter}/{total_pairs}")

def get_score_sift(file_path, json_file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('Valid answer:')
            distance = float(parts[0].split(':')[1].strip())
            result = 1 if distance >= 90 else 0
            results.append(result)


    with open(json_file_path, 'r') as file:
        data = json.load(file)

    answers = []
    for result in data['data']['results']:
        answer = result['answers'][0]['answer'][0]['id']
        answers.append(int(answer))

    f1 = f1_score(results, answers)
    return f1

def get_score(file_path_1, file_path_2, json_file_path):
    results1 = []
    results2 = []
    with open(file_path_1, 'r') as file:
        for line in file:
            parts = line.split('Valid answer:')
            distance = float(parts[0].split(':')[1].strip())
            result1 = 1 if distance >= 140 else 0
            results1.append(result1)

    with open(file_path_2, 'r') as file:
        for line in file:
            parts = line.split('Valid answer:')
            distance = float(parts[0].split(':')[1].strip())
            result2 = 1 if distance >= 140 else 0
            results2.append(result2)

    results = np.add(results1, results2)
    results = np.where(results == 2, 1, 0)

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    answers = []
    for result in data['data']['results']:
        answer = result['answers'][0]['answer'][0]['id']
        answers.append(int(answer))

    f1 = f1_score(results, answers)
    return f1

def download_image(image_url, output_path):
    try:
        urllib.request.urlretrieve(image_url, output_path)
        print(f"Downloaded {image_url}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")

def download_images(json_path, download_dir):
    with open(json_path, 'r') as file:
        data = json.load(file)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    counter = 0
    for result in data['data']['results']:
        task_id = result['taskId']
        image1_url = result['representativeData']['image1']['imageUrl']
        image2_url = result['representativeData']['image2']['imageUrl']

        image1_name = f"{task_id}_1.jpg"
        image2_name = f"{task_id}_2.jpg"

        image1_path = os.path.join(download_dir, image1_name)
        image2_path = os.path.join(download_dir, image2_name)

        download_image(image1_url, image1_path)
        download_image(image2_url, image2_path)

        counter += 1
        print(counter)

def main():
    json_path_train = 'C:\\Users\\acer\\Desktop\\Week2\\date\\train_task1.json'
    json_path_test = 'C:\\Users\\acer\\Desktop\\Week2\\date\\test_task1.json'
    json_path_val = 'C:\\Users\\acer\\Desktop\\Week2\\date\\val_task1.json'
    result_path_train_sift = 'C:\\Users\\acer\\Desktop\\Week2\\results\\output_train_results_sift.txt'
    result_path_train_orb = 'C:\\Users\\acer\\Desktop\\Week2\\results\\output_train_results_orb.txt'
    result_path_test_sift = 'C:\\Users\\acer\\Desktop\\Week2\\results\\output_test_results_sift.txt'
    result_path_test_orb = 'C:\\Users\\acer\\Desktop\\Week2\\results\\output_test_results_orb.txt'
    images_folder = 'E:\\images_rooms'
    compare_images_csv(json_path_val, images_folder)
    #f1 = get_score_sift(result_path_train_sift, json_path_train)
    #print(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()