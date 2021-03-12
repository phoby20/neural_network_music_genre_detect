import numpy
import pandas as pd
import scipy.special
import pickle
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import json
import os.path


class neuralNetwork:

    # ネットワークの初期化
    def __init__(self, inputnodes, hiddennodes, ouputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = ouputnodes

        # 学習率
        self.lr = learningrate


        #  -------------------------------------------------------------------------------------------------------------
        self.who_path = 'learn_json/who.json'
        self.wih_path = 'learn_json/wih.json'



        if os.path.isfile(self.wih_path):
            # 저장한 가중치 오픈하기
            wih_json_open = open(self.wih_path, 'r')
            wih_json_load = json.load(wih_json_open)
            wih_json_open.close()

            # 저장한 가중치 할당하기
            wih_array = numpy.asarray(wih_json_load)
            self.wih = wih_array

        else:
            # 가중치 초기화 하기
            # 0을 중심으로 1/√(들어오는 연결 노드의 갯수)의 표준편차
            # numpy.random.normal(정규분포의 중심, 표준편차, numpy행렬)
            self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        if os.path.isfile(self.who_path):
            # 저장한 가중치 오픈하기
            who_json_open = open(self.who_path, 'r')
            who_json_load = json.load(who_json_open)
            who_json_open.close()
            # print('who_json_load:', who_json_load)

            # 저장한 가중치 할당하기
            who_array = numpy.asarray(who_json_load)
            self.who = who_array
            # print('who_array:', who_array)

        else:
            # 가중치 초기화 하기
            # 0을 중심으로 1/√(들어오는 연결 노드의 갯수)의 표준편차
            # numpy.random.normal(정규분포의 중심, 표준편차, numpy행렬)
            self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #  -------------------------------------------------------------------------------------------------------------



        # 가중치 초기화 하기
        # 0을 중심으로 1/√(들어오는 연결 노드의 갯수)의 표준편차
        # numpy.random.normal(정규분포의 중심, 표준편차, numpy행렬)
        # self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 学習
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)  # 은닉계층으로 들어오는 신호를 계산 (인풋 가중치 행렬 * 입력된 행렬)
        hidden_outputs = self.activation_function(hidden_inputs)  # 은닉계층에서 나가는 신호를 계산 (위에서 계산된 행렬를 시그모이드 함수로 계산)
        final_inputs = numpy.dot(self.who, hidden_outputs)  # 최종 출력 계층으로 들어오는 신호를 계산 (아웃풋 가중치 행렬 * 위에서 계산된 행렬)
        final_outputs = self.activation_function(final_inputs)  # 최종 출력 계층에서 나가는 신호를 계산 (위에서 계산된 행렬를 시그모이드 함수로 계산)

        output_errors = targets - final_outputs  # 실제 값과 계산 결과의 오차값 (실제 값 - 결과 값)
        hidden_errors = numpy.dot(self.who.T, output_errors)  # 은닉계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))  # 은닉계층과 출력계층간의 가중치 업데이트
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))  # 입력계층과 은닉계층 간의 가중치 업데이트

        pass

    # ネットワークに質疑
    def query(self, inputs_list):
        # 입력 신호를 2차원의 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 은닉계층으로 들어오는 신호를 계산 (인풋 가중치 행렬 * 입력된 행렬)
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 은닉계층에서 나가는 신호를 계산 (위에서 계산된 행렬를 시그모이드 함수로 계산)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 최종 출력 계층으로 들어오는 신호를 계산 (아웃풋 가중치 행렬 * 위에서 계산된 행렬)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산 (위에서 계산된 행렬를 시그모이드 함수로 계산)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
