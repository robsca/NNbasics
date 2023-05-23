'''
To run the program, open the terminal and type:

streamlit run main.py

'''
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("Hello World")

def prime_number_checker(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


n = st.number_input(
    label = "Enter a number", 
    min_value = 100,
    max_value = 100000,
    value = 100,
    step = 1
)

result = prime_number_checker(n)

if result:
    st.success(f"{n} is a prime number")

def NeuralNetwork(data):
    '''
    Goal:
    This network needs to predict the next prime number.

    Approach this problem:
    1. Create a Neural Network -> that takes input and output

    input = n
    output = next prime number

    '''
    # take off the first row
    data = data.iloc[1:]
    # the Neural Net needs examples of the data to learn from
    features = ['number', 'is_a_even_number', 'is_a_odd_number'] 
    input = data[features]
    output = data['distance']

    for i, row in data.iterrows():
        input_row = row[features]
        output_row = row['distance']
        st.write(input_row)
        st.write(output_row)

    # divide the data intp training set and test set
    # 80% training set
    # 20% test set






def creating_a_dataset(number_of_primes = 100):
    '''
    Table of prime numbers
    ---
    columns = ['prime_number']
    '''

    # 1. Create an empty dataframe to store the prime numbers
    columns = ['number', 'prime_number', 'is_a_even_number', 'is_a_odd_number']
    df = pd.DataFrame( columns=columns)

    n = 1
    while len(df) < number_of_primes:
        # 3. Check if the number is a prime number
        result = prime_number_checker(n)

        # 4. Creating a row for the dataframe
        df = df.append({
                'number': n, # 1 integer
                'prime_number': result, # boolean

                'is_a_even_number': n % 2 == 0,
                'is_a_odd_number': n % 2 != 0

            }, ignore_index = True)
        
        # 5. Increment the number
        n += 1

    # modify data to keep only prime numbers
    df = df[df['prime_number'] == True]
    # calculate the distance between the prime numbers
    df['distance'] = df['number'].diff()
    # reset the index
    df = df.reset_index(drop=True)

    return df  


data = creating_a_dataset(n)

with st.expander("Show the data"):
    st.write(data)


# create a nother graeph that show only the prime numbers
data_prime = data[data['prime_number'] == True]
# reset the index
data_prime = data_prime.reset_index(drop=True)


# only the prime numbers
fig_prime = px.scatter(
    data_prime,
    x = data_prime.index,
    y = 'number')

st.plotly_chart(fig_prime)


NeuralNetwork(data)




