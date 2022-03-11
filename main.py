
import math
import random


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
	'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
layer_counts = [26, 30, 10, 5, 5]


class Wordle:

	def __init__(self, init, pop_size) -> None:
		if(init == 'random'):
			self.word = random.choice(self.grabFile('words.txt').split(','))
		else:
			self.word = init

		self.p = Population(pop_size)

	def bestFitness(self):
		return self.p.highestFitness()

	def grabFile(self, path) -> str:
		with open(path, 'r') as f:
			return f.read()

	def getWord(self) -> str:
		return self.word

	def prove(self, ammount):
		for i in range(1, ammount + 1):
			self.step(i)

	def step(self, s):
		print("Generation #" + str(s))
		for i in range(6):
			self.p.cycle(self, i + 1)

	def generateKeyMap(self, guess):  # guess is a 5 letter string

		current_word = self.word

		if(guess == current_word):
			return 1

		'''
		# Convert Letters into Numbers ??

		# Convert the guess into a list of 5 numbers:

		# a = 0, b = 1, c = 2, d = 3 …



		numeric_guess = ~~ Magic way of converting above ~~
		'''

		new_keymap = [0] * 26

		for i, letter in enumerate(current_word):

			# don’t overwrite information we know

			if(guess[i] == letter):

				new_keymap[letters.index(guess[i])] = max(
					new_keymap[letters.index(guess[i])], 3)

			elif(guess[i] in current_word):

				new_keymap[letters.index(guess[i])] = max(
					new_keymap[letters.index(guess[i])], 2)

			elif(guess[i] not in current_word):

				new_keymap[letters.index(guess[i])] = max(
					new_keymap[letters.index(guess[i])], 1)

		# a list of length 26 and has individual values of [0, 3]
		return new_keymap


class NeuralNetwork:

	def __init__(self, layer_counts=[], init=False):

		self.layers = []
		self.layer_counts = layer_counts

		if(init):
			self.initNN()

	def initNN(self):

		for i in range(len(self.layer_counts) - 1):
			layer = []
			for j in range(self.layer_counts[i] * self.layer_counts[i + 1]):
				layer.append(random.random() * 2 - 1)

			self.layers.append(layer)
		

	def pushForward(self, input):  # input is kepmap of current unit

		for i in range(1, len(self.layers)):

			next_layer_size = len(self.layers[i]) / self.layer_counts[i+1]
			new_input = []
			for j in range(int(next_layer_size)):
				sum = 0
				for k in range(self.layer_counts[i+1]):
					sum += self.layers[i][j *
						self.layer_counts[i+1] + k] * input[k]
				new_input.append(self.sigmoid(sum))
			input = new_input
		self.output = input
		return input

	def sigmoid(self, x) -> float:
		return 1 / (1 + math.exp(-x))

	def getOutputGuessWord(self, keymap) -> str:
		self.pushForward(keymap)
		output = self.output  # output should be a list of 5 numbers [0, 1]

		string = ''
		for o in output:  # o is just a number [0, 1]
			mapped = math.floor(self.map(o, 0, 1, 0, 26))
			string += letters[mapped]
		return string

	def map(self, x, a, b, c, d) -> float:
		return (x - a) / (b - a) * (d - c) + c

	def mergeLayers(self, partner, mutate=False):
		mutation_strength = 0.01
		new_layers = []
		
		for i in range(len(self.layer_counts) - 1):
			layer = [0] * self.layer_counts[i] * self.layer_counts[i + 1]
			for j in range(self.layer_counts[i] * self.layer_counts[i + 1]):
				
				partition = random.random()
				if(partition < 0.5):
					if(mutate):
						layer[j] = partner.brain.layers[i][j] + (random.random() * 2 - 1) * mutation_strength
					else:
						layer[j] = partner.brain.layers[i][j]
				else:
					if(mutate):
						layer[j] = self.layers[i][j] + (random.random() * 2 - 1) * mutation_strength
					else:
						layer[j] = self.layers[i][j]

			new_layers.append(layer)
				
		return new_layers
		

	def setLayers(self, layers) -> None:
		self.layers = layers


class Unit:

	def __init__(self, ID, keymap=[], network=None):  # Don’t know which arg should go first??
		self.ID = ID
		self.correct = False
		self.current_word = ''
		self.fitness = 0

		if(len(keymap) == 0):
			self.keymap = [0] * 26
		else:
			self.keymap = keymap

		if(network):
			self.brain = network
		else:
			self.brain = NeuralNetwork(layer_counts, init=True)

	def __repr__(self) -> str:
		return str(self.current_word) + ", " + str(len(self.brain.layers[0])) + ", " + str(len(self.brain.layers[1])) + ", " + str(len(self.brain.layers[2])) + ", " + str(len(self.brain.layers[3]))

	def guessWord(self):
		word = self.brain.getOutputGuessWord(self.keymap)
		self.current_word = word
		return word


	def calculateFitness(self, guess_count):
		if(self.correct):
			self.fitness = 1
		else:
			keymap_ = sorted(self.keymap, reverse=True)
			for i in range(len(self.current_word)):
				self.fitness += keymap_[i]

			self.fitness /= 15
			return self.fitness

	def singleCycle(self, wordle, guess_count):
		guess = self.guessWord()
		km = wordle.generateKeyMap(guess)
		if(km == 1):
			self.correct = True
		else:
			self.keymap = km
			fitness = self.calculateFitness(guess_count)
			return fitness
		return True

	

	'''
	
	Problem is here VVV
	
	'''

	def merge(self, partner, mutate = False):
		n = NeuralNetwork(layer_counts , init = True)
		n.layers = self.brain.mergeLayers(partner, mutate)
		#n.setLayers(n.mergeLayers(partner, mutate))
		u = Unit(0, [], n)
		return u







class Population:

	def __init__(self, size):
		self.global_id = 0
		self.size = size
		self.pool = [Unit(self.getGlobalID(), [], None) for i in range(size)] # [] and None are not needed

	def highestFitness(self, pool):
		highest = 0
		for u in pool:
			if(u.fitness > highest):
				highest = u.fitness
		return highest

	def averageFitness(self, pool):
		total = 0
		for u in pool:
			total += u.fitness
		return total / len(pool)

	def getGlobalID(self):
		self.global_id += 1
		return self.global_id

	def cycle(self, wordle, guess_count):
		pass_pool = []
		print_pool = []

		for p in self.pool:
			fitness = p.singleCycle(wordle, guess_count)
			pass_pool.append([fitness, p])
			print_pool.append(p)

		print("Average fitness: " + str(self.averageFitness(print_pool)))
		breeding_pool = self.generateBreedingPool(pass_pool)
		print("Average fitness: " + str(self.averageFitness(breeding_pool)))
		bred_pool = self.breed(breeding_pool)
		print("Average fitness: " + str(self.averageFitness(bred_pool)))
		self.pool = self.pruneBreedingPool(bred_pool)

		print("Guess count: " + str(guess_count))
		print("Highest fitness: " + str(self.highestFitness(self.pool)))
		print("Average fitness: " + str(self.averageFitness(self.pool)))


	def generateBreedingPool(self, pass_pool): # !! len(pass_pool) = len(self.pool)

		# if their fitness is true it means they have already guessed the word and can be included in the final pool the max number of times possible
		# otherwise it will be a number [0, 1] and we can scale the pool accordingly 
		# This function generates a breeding pool that is larger than self.pool and has a majority of well performing units and a minority of non-well performing 		units

		breeding_pool = []

		for i in range(len(pass_pool)):
			# if pass_pool[i][0] > .8:
			# 	print("Fitness:", pass_pool[i][0], "Unit:", pass_pool[i][1].current_word)
			for j in range(int(int(pass_pool[i][0] * 10) ** 1.2)):
				breeding_pool.append(pass_pool[i][1])
		
		return breeding_pool


	def breed(self, breeding_pool):

		new_pool = []	
		left_pool = []
		right_pool = []

		# Shuffle breeding pool
		random.shuffle(breeding_pool)


		for i, unit in enumerate(breeding_pool):
			if(i % 2 == 0):
				left_pool.append(unit)
			else:
				right_pool.append(unit)

		# force left_pool and right_pool to be the same length
		if(len(left_pool) > len(right_pool)):
			left_pool.pop()
		elif(len(right_pool) > len(left_pool)):
			right_pool.pop()
		else:
			pass

		for i in range(len(left_pool)):
			m = left_pool[i].merge(right_pool[i], True)
			print(m)
			new_pool.append(m)

		
		return new_pool



	def pruneBreedingPool(self, breeding_pool):

			# breeding_pool is larger than 200 but not much larger otherwise threshold_inc will be too small
			new_pool = [] 
			new_pool_max_size = 200
			threshold = 0
			threshold_inc = 1 / len(breeding_pool)  	
			for unit in breeding_pool:
				
				if(len(new_pool) == new_pool_max_size):
					return new_pool

				target = random.random() # 0-1

				if(target > threshold):
					new_pool.append(unit)
					threshold += threshold_inc


			count = new_pool_max_size - len(new_pool)		
			new_units = [Unit(self.getGlobalID(), [], NeuralNetwork(layer_counts, init = True)) for i in range(count)]
			new_pool.extend(new_units)
			return new_pool 













w = Wordle('heron', 200)
w.prove(5)