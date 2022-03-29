//
//  Genetic-Model.swift
//  ANN-applied-to-Genetic-Algorithm
//
//  Created by Eugenio Raja on 26/03/22.
//

import Foundation
import Combine

class Vector: ObservableObject {
    @Published public var x: Float
    @Published public var y: Float
    
    init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
    
    init(vector: Vector) {
        self.x = vector.x
        self.y = vector.y
    }
    
    init(vector: Vector, x: Float, y: Float) {
        self.x = vector.x + x
        self.y = vector.y + y
    }
    
    func add(_ vector: Vector) {
        self.x += vector.x
        self.y += vector.y
    }
    
    func limit(_ max: Float) {
        let magnitudeSquared = (self.x * self.x) + (self.y * self.y);
        let maxSquared = max * max;
        if (magnitudeSquared <= maxSquared) {
            return;
        }

        let magnitude = sqrt(magnitudeSquared);
        let normalizedX = self.x / magnitude;
        let normalizedY = self.y / magnitude;
        let newX = normalizedX * max;
        let newY = normalizedY * max;

        self.x = newX;
        self.y = newY;
    }
    
    func distance(_ other: Vector) -> Float {
        let diffX = self.x - other.x;
        let diffY = self.y - other.y;

        let magnitudeSquared = (diffX * diffX) + (diffY * diffY);
        let magnitude = sqrt(magnitudeSquared);

        return magnitude;
    }
}

class Brain {
    public var network = NeuralNetwork(
        layers: [
            Dense(inputSize: 2, neuronsCount: 4, functionRaw: .sigmoid),
            Dense(inputSize: 4, neuronsCount: 4, functionRaw: .sigmoid),
            Dense(inputSize: 4, neuronsCount: 4, functionRaw: .sigmoid)
        ], learningRate: 0.5, epochs: 30, batchSize: 16)
    
    public var set = Dataset(items: [])
    
    public let size: Int
    
    public var step: Int
    
    init(size: Int, dataset: Dataset?) {
        self.size = size
        self.step = 0
        if (dataset != nil) {
            self.set = dataset!
        }
        else { }
    }
    
    init(copy: Brain) {
        self.size = copy.size
        self.step = 0
    }
    
}

class Dot: Identifiable, ObservableObject {
    public let id = UUID()
    
    @Published public var dead: Bool
    @Published public var success: Bool
    @Published public var champion: Bool
    @Published public var position: Vector;
    public let brain: Brain;
    
    private let velocity: Vector
    private var acceleration: Vector;
    private let maxWidth: Int;
    private let maxHeight: Int;
    private let minTargetDistance: Float;
    private let dotSize: Int
    
    init(width: Int, height: Int, dotSize: Int, minTargetDistance: Float, brain: Brain) {
        self.velocity = Vector(x: 0, y: 0)
        self.acceleration = Vector(x: 0, y: 0)
        self.position = Vector(x: Float(width / 2), y: Float(height - 10))
        self.dead = false
        self.success = false
        self.maxWidth = width
        self.maxHeight = height
        self.minTargetDistance = minTargetDistance
        self.dotSize = dotSize
        self.champion = false
        
        self.brain = brain
        let _ = self.brain.network.train(set: self.brain.set)
    }
    
    init(copy: Dot) {
        self.velocity = Vector(x: 0, y: 0)
        self.acceleration = Vector(x: 0, y: 0)
        self.position = Vector(x: Float(copy.maxWidth / 2), y: Float(copy.maxHeight - 10))
        self.dead = false
        self.success = false
        self.maxWidth = copy.maxWidth
        self.maxHeight = copy.maxHeight
        self.minTargetDistance = copy.minTargetDistance
        self.dotSize = copy.dotSize
        self.champion = true
        self.brain = Brain(copy: copy.brain)
        let _ = self.brain.network.train(set: self.brain.set)
    }
    
    func update(target: Vector) {
        let halfDot = Float(self.dotSize) / 2.0;
        let maxX = Float(self.maxWidth) - halfDot;
        let maxY = Float(self.maxHeight) - halfDot;

        if (self.dead || self.success) {
            return;
        }

        self.move()
        self.objectWillChange.send()
        
        if (self.brain.step >= self.brain.size) {
            self.dead = true
            return
        }
        
        if (self.position.x <= halfDot || self.position.x >= maxX) {
            self.dead = true
            return
        }

        if (self.position.y <= halfDot || self.position.y >= maxY) {
            self.dead = true
            return
        }

        let distance = self.position.distance(target)
        if (distance < minTargetDistance) {
            self.success = true
            return
        }
    }
    
    func fitness(target: Vector) -> Float {
        if (self.success) {
            let size = Float(self.dotSize)
            let step = Float(self.brain.step)
            let targetFitness = 1.0 / (size * size)
            let stepFitness = 10000.0 / (step * step)
            let fitness = targetFitness + stepFitness

            return fitness;
        } else {
            let distance = self.position.distance(target);
            let fitness = 1.0 / (distance * distance);

            return fitness;
        }
    }
    
    private func move() {
        
        let prediction = self.brain.network.predict(input: .init(size: .init(width: 2), body: [self.position.x, self.position.y]))

        let goLeft = round(prediction[0])
        let goUp = round(prediction[1])
        let goRight = round(prediction[2])
        let goDown = round(prediction[3])

        let dir = Vector(x: (goRight - goLeft) * 50, y: (goDown - goUp) * 50)
        
        self.acceleration = dir
        self.velocity.add(self.acceleration)
        self.velocity.limit(5)
        self.position.add(self.velocity)
        self.brain.step += 1
        if (self.brain.step % 75 == 0) {
            self.brain.set.items.append(
                .init(input: .init(size: .init(width: 2), body: [self.position.x, self.position.y]),
                      output: .init(size: .init(width: 4), body: [prediction[0], prediction[1], prediction[2], prediction[3]]))
            )
        }
    }
}

class Population: ObservableObject {
    @Published public var dots: [Dot] = []
    @Published public var generation: Int
    @Published public var minStep: Int
    
    private let size: Int
    private let width: Int
    private let height: Int
    private let dotSize: Int
    private let brainSize: Int
    private let minTargetDistance: Float
    private let mutationRatio: Float
    
    init(
        populationSize: Int,
        width: Int,
        height: Int,
        dotSize: Int,
        brainSize: Int,
        minTargetDistance: Float,
        mutationRatio: Float
    ) {
        self.size = populationSize
        self.width = width
        self.height = height
        self.dotSize = dotSize
        self.brainSize = brainSize
        self.minTargetDistance = minTargetDistance
        self.mutationRatio = mutationRatio
        self.minStep = brainSize
        self.generation = 1
        
        self.generate(brainSize: self.brainSize);
    }
    
    func update(target: Vector) {
        self.dots.forEach { $0.update(target: target) }
    }
    
    func naturalSelection(target: Vector) {
        
        var fitnessSum: Float = 0;
        var maxFitness: Float = -1;
        var maxFitnessIndex = -1;
        var minSteps = -1;

        for index in 0..<dots.count {
            let dot = self.dots[index];
            let fitness = dot.fitness(target: target);
            if (minSteps == -1) {
                minSteps = dot.brain.size
            }

            if (fitness > maxFitness) {
                maxFitness = fitness
                maxFitnessIndex = index

                if (dot.success) {
                    minSteps = min(minSteps, dot.brain.step)
                }
            }

            fitnessSum += fitness;
        }

        let champion = self.dots[maxFitnessIndex];
        self.minStep = minSteps
        
        let fittestValue = Float.random(in: 0...1) * fitnessSum;
        var fittest: Dot? = nil;
        var runningSum: Float = 0;
        for index in 0..<dots.count {
            runningSum += self.dots[index].fitness(target: target);
            if (runningSum >= fittestValue) {
                fittest = self.dots[index];
                break;
            }
        }

        var newDots: [Dot] = [];
        newDots.append(Dot(copy: champion));
        let dataset = champion.brain.set
        for _ in 1..<dots.count {
            var newdataset = Dataset(items: [])
            for i in 0..<dataset.items.count {
                if Float.random(in: 0...1) >= self.mutationRatio {
                    newdataset.items.append(dataset.items[i])
                } else {
                    newdataset.items.append(
                        .init(input: .init(size: .init(width: 2), body: [Float.random(in: 0...700), Float.random(in: 0...700)]),
                              output: .init(size: .init(width: 4), body: [round(Float.random(in: 0...1)),
                                                                          round(Float.random(in: 0...1)),
                                                                          round(Float.random(in: 0...1)),
                                                                          round(Float.random(in: 0...1))]))
                    )
                }
            }
            let nextDot = Dot(
                width: self.width,
                height: self.width,
                dotSize: self.dotSize,
                minTargetDistance: minTargetDistance,
                brain: Brain(size: self.brainSize, dataset: newdataset)
            )
            
            newDots.append(nextDot)
        }

        self.dots.removeAll()
        self.dots.append(contentsOf: newDots)

        self.generation += 1
    }
    
    func allDead() -> Bool {
        for dot in dots {
            if !dot.dead && !dot.success {
                return false
            }
        }
        
        return true
    }
    
    private func generate(brainSize: Int) {
        self.dots.removeAll()
        
        var newDots: [Dot] = []
        for _ in 0..<size {
            
            let brain = Brain(size: brainSize, dataset: nil)
            let dot = Dot(
                width: width,
                height: height,
                dotSize: dotSize,
                minTargetDistance: minTargetDistance,
                brain: brain
            )

            newDots.append(dot)
        }
        
        self.dots.append(contentsOf: newDots)
    }
}
