//
//  ANN-Model.swift
//  ANN-applied-to-Genetic-Algorithm
//
//  Created by Eugenio Raja on 26/03/22.
//

import Foundation

public enum DataSizeType: Int, Codable {
    case oneD = 1
    case twoD
    case threeD
}

public struct DataSize: Codable {
    var type: DataSizeType
    var width: Int
    var height: Int?
    var depth: Int?
    
    public init(width: Int) {
        type = .oneD
        self.width = width
    }
    
    public init(width: Int, height: Int) {
        type = .twoD
        self.width = width
        self.height = height
    }
    
    public init(width: Int, height: Int, depth: Int) {
        type = .threeD
        self.width = width
        self.height = height
        self.depth = depth
    }
    
}

public struct DataPiece: Equatable {
    public static func == (lhs: DataPiece, rhs: DataPiece) -> Bool {
        return lhs.body == rhs.body
    }
    
    public var size: DataSize
    public var body: [Float]
    
    func get(x: Int) -> Float {
        return body[x]
    }
    
    func get(x: Int, y: Int) -> Float {
        return body[x+y*size.width]
    }
    
    func get(x: Int, y: Int, z: Int) -> Float {
        return body[z+(x+y*size.width)*size.depth!]
    }
    
    public init(size: DataSize, body: [Float]) {
        var flatSize = size.width
        if let height = size.height {
            flatSize *= height
        }
        if let depth = size.depth {
            flatSize *= depth
        }
        if flatSize != body.count {
            fatalError("DataPiece body does not conform to DataSize.")
        }
        self.size = size
        self.body = body
    }
    
}

public struct DataItem {
    var input: DataPiece
    var output: DataPiece
    
    public init(input: DataPiece, output: DataPiece) {
        self.input = input
        self.output = output
    }
    
    public init(input: [Float], inputSize: DataSize, output: [Float], outputSize: DataSize) {
        self.input = DataPiece(size: inputSize, body: input)
        self.output = DataPiece(size: outputSize, body: output)
    }
}

public struct Dataset {
    public var items: [DataItem]
    
    public init(items: [DataItem]) {
        self.items = items
    }
    
}

final public class NeuralNetwork {
    public var layers: [Layer] = []
    public var learningRate: Float
    public var epochs: Int
    public var batchSize: Int
    
    public init(layers: [Layer], learningRate: Float, epochs: Int, batchSize: Int) {
        self.layers = layers
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
    }
    
    public func printSummary() {
        for layer in layers {
            print("Dense layer: \(layer.neurons.count) neurons")
        }
    }
    
    public func train(set: Dataset) -> Float {
        var error = Float.zero
        for epoch in 0..<epochs {
            var shuffledSet = set.items.shuffled()
            error = Float.zero
            while !shuffledSet.isEmpty {
                let batch = shuffledSet.prefix(batchSize)
                for item in batch {
                    let predictions = forward(networkInput: item.input)
                    for i in 0..<item.output.body.count {
                        error+=pow(item.output.body[i]-predictions.body[i], 2)/2
                    }
                    backward(expected: item.output)
                    deltaWeights(row: item.input)
                }
                for layer in layers {
                    layer.updateWeights()
                }
                shuffledSet.removeFirst(min(batchSize,shuffledSet.count))
            }
//            print("Epoch \(epoch+1), error \(error).")
        }
        return error
    }
    
    public func predict(input:DataPiece) -> [Float] {
        return forward(networkInput: input).body
    }
    
    private func deltaWeights(row: DataPiece) {
        var input = row
        for i in 0..<layers.count {
            input = layers[i].deltaWeights(input: input, learningRate: learningRate)
        }
    }
    
    private func forward(networkInput: DataPiece) -> DataPiece {
        var input = networkInput
        for i in 0..<layers.count {
            input = layers[i].forward(input: input)
        }
        return input
    }
    
    private func backward(expected: DataPiece) {
        var input = expected
        var previous: Layer? = nil
        for i in (0..<layers.count).reversed() {
            input = layers[i].backward(input: input, previous: previous)
            previous = layers[i]
        }
    }
}

struct Neuron {
    var weights: [Float]
    var weightsDelta: [Float]
    var bias: Float
    var delta: Float
}

public func classifierOutput(classes: Int, correct: Int) -> DataPiece {
    if correct>=classes {
        fatalError("Correct class must be less than classes number.")
    }
    var output = Array(repeating: Float.zero, count: classes)
    output[correct] = 1.0
    return DataPiece(size: .init(width: classes), body: output)
}

public class Layer {
    var neurons: [Neuron] = []
    var function: ActivationFunction
    var output: DataPiece?
    
    func forward(input: DataPiece) -> DataPiece {
        return input
    }
    
    func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        return input
    }
    
    func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return input
    }
    
    func updateWeights() {
        return
    }
    
    init(function: ActivationFunction) {
        self.function = function
    }
}

public class Dense: Layer {
    
    private let queue = DispatchQueue.global(qos: .userInitiated)
    
    public init(inputSize: Int, neuronsCount: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        super.init(function: function)
        output = .init(size: .init(width: neuronsCount), body: Array(repeating: Float.zero, count: neuronsCount))
        
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons.append(Neuron(weights: weights, weightsDelta: .init(repeating: Float.zero, count: weights.count), bias: 0.0, delta: 0.0))
        }
    }
    
    override func forward(input: DataPiece) -> DataPiece {
        input.body.withUnsafeBufferPointer { inputPtr in
            output?.body.withUnsafeMutableBufferPointer { outputPtr in
                neurons.withUnsafeBufferPointer { neuronsPtr in
                    DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                        var out = neuronsPtr[i].bias
                        neuronsPtr[i].weights.withUnsafeBufferPointer { weightsPtr in
                            DispatchQueue.concurrentPerform(iterations: neuronsPtr[i].weights.count, execute: { i in
                                out += weightsPtr[i] * inputPtr[i]
                            })
                        }
                        outputPtr[i] = function.activation(input: out)
                    })
                }
            }
        }
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = Array(repeating: Float.zero, count: neurons.count)
        if let previous = previous {
            for j in 0..<neurons.count {
                for neuron in previous.neurons {
                    errors[j] += neuron.weights[j]*neuron.delta
                }
            }
        } else {
            for j in 0..<neurons.count {
                errors[j] = input.body[j] - output!.body[j]
            }
        }
        for j in 0..<neurons.count {
            neurons[j].delta = errors[j] * function.derivative(output: output!.body[j])
        }
        return output!
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            input.body.withUnsafeBufferPointer { inputPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        DispatchQueue.concurrentPerform(iterations: deltaPtr.count, execute: { j in
                            deltaPtr[j] += learningRate * neuronsPtr[i].delta * inputPtr[j]
                        })
                        neuronsPtr[i].bias += learningRate * neuronsPtr[i].delta
                    }
                })
            }
        }
        return output!
    }
    
    override func updateWeights() {
        let neuronsCount = neurons.count
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            DispatchQueue.concurrentPerform(iterations: neuronsCount, execute: { i in
                neuronsPtr[i].weights.withUnsafeMutableBufferPointer { weightsPtr in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        let weightsCount = deltaPtr.count
                        DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { j in
                            weightsPtr[j] += deltaPtr[j]
                            deltaPtr[j] = 0
                        })
                    }
                }
            })
        }
    }
    
}

func getActivationFunction(rawValue: Int) -> ActivationFunction {
    switch rawValue {
    default:
        return Sigmoid()
    }
}

public enum ActivationFunctionRaw: Int {
    case sigmoid = 0
}

protocol ActivationFunction {
    var rawValue: Int { get }
    func activation(input: Float) -> Float
    func derivative(output: Float) -> Float
}

struct Sigmoid: ActivationFunction {
    public var rawValue: Int = 0
    
    public func activation(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
    
    public func derivative(output: Float) -> Float {
        return output * (1.0-output)
    }
}
