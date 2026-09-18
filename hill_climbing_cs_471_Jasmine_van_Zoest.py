import random # needed for random-restart hill climbing 


def f(x): # function to maximize, f(x) = 2 - x^2, domain [-5, 5]
    return 2 - x**2


def g(x): # g(x) = 0.0051x^5 - 0.1367x^4 + 1.24x^3 - 4.456x^2 + 5.66x - 0.287, domain [0, 10]
    return (
        0.0051 * x**5
        - 0.1367 * x**4
        + 1.24 * x**3
        - 4.456 * x**2
        + 5.66 * x
        - 0.287
    )


def create_state_space(lower, upper, step): # creates a list of discrete states so the hill climbing algorithm can properly evaluate neighbors
    #cacluates the steps in the range
    number_of_steps = (upper - lower) / step

    if not number_of_steps.is_integer():
        raise ValueError("The step size must divide the range exactly.")

    return [
        round(lower + index * step, 10) #a list of discrete states from lower to upper with the given step size
        for index in range(int(number_of_steps) + 1)
    ]



def get_neighbors(x, states): #returns the neighbors of the state x (the previous and next states)
    index = states.index(x)
    neighbors = []

    if index > 0:
        neighbors.append(states[index - 1])
    if index < len(states) - 1:
        neighbors.append(states[index + 1])

    return neighbors


def hill_climbing(function, states, start_x): #performs the algorithm, finds the maximum value of function
    current_x = start_x

    while True: #tests the neighbors of the current state and moves to the neighbor with the highest value if it is greater than the current state
        current_value = function(current_x)
        neighbors = get_neighbors(current_x, states)

        if not neighbors:
            return current_x, current_value

        next_x = max(neighbors, key=function)
        next_value = function(next_x)

        if next_value <= current_value:
            return current_x, current_value

        current_x = next_x


def random_restart_hill_climbing(function, states, restarts=20): #random-restart hill climbing
    best_x = None
    best_value = float("-inf") #float so it can be compared to the function values

    for _ in range(restarts):
        start_x = random.choice(states)
        local_x, local_value = hill_climbing(function, states, start_x)

        if local_value > best_value: #ensures that the best value is updated if a better value is found
            best_x = local_x
            best_value = local_value

    return best_x, best_value


def main(): #runs the program, prompts the user for input, and displays the results
    print("Choose a function:")
    print("1. f(x) = 2 - x^2, domain [-5, 5]")
    print("2. g(x), domain [0, 10]")

    function_choice = input("Enter 1 or 2: ").strip() #promts the user to choose between the two functions to maximize

    if function_choice == "1": #f(x) = 2 - x^2, domain [-5, 5]
        function = f
        lower_bound = -5
        upper_bound = 5
    elif function_choice == "2": #g(x), domain [0, 10]
        function = g
        lower_bound = 0
        upper_bound = 10
    else:
        print("Invalid function choice.") #incase the user inputs something other than 1 or 2
        return

    try:
        step = float(input("Enter step size, for example 0.5: ")) #prompts the user to enter a step size
        if step <= 0:
            raise ValueError("The step size must be greater than zero.")
        states = create_state_space(lower_bound, upper_bound, step)
    except ValueError as error:
        print(f"Invalid input: {error}")
        return

    print("\nChoose an algorithm:")
    print("1. Standard hill climbing")
    print("2. Random-restart hill climbing")
    algorithm_choice = input("Enter 1 or 2: ").strip()
    #prompts the user to choose an algorithm

    if algorithm_choice == "1":
        #performs standard hill climbing
        start_x = round(
            float(input(f"Enter starting x between {lower_bound} and {upper_bound}: ")),
            10,
        )

        if start_x not in states:
            print("Starting x must be one of the discrete states.")
            return

        best_x, best_value = hill_climbing(function, states, start_x)

    elif algorithm_choice == "2":
        #performs random-restart hill climbing
        restarts_input = input("Enter number of restarts [default: 20]: ")
        restarts = int(restarts_input) if restarts_input else 20

        if restarts <= 0:
            print("The number of restarts must be greater than zero.")
            return

        best_x, best_value = random_restart_hill_climbing(
            function,
            states,
            restarts,
        )
    else:
        print("Invalid algorithm choice.")
        return

    print("\nResult")
    print(f"Best x-value: {best_x}")
    print(f"Maximum value: {best_value}")


if __name__ == "__main__":
    main()
