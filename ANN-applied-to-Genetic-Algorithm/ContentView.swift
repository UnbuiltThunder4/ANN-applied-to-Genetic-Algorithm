//
//  ContentView.swift
//  ANN-applied-to-Genetic-Algorithm
//
//  Created by Eugenio Raja on 26/03/22.
//

import SwiftUI

struct DotView: View {
    @ObservedObject var dot: Dot
    let dotSize: CGFloat
    
    init(_ dot: Dot, _ size: CGFloat) {
        self.dot = dot;
        self.dotSize = size
    }
    
    var body: some View {
        ZStack {
            Rectangle()
                .foregroundColor(self.dotColor(dot: dot))
                .frame(width: self.dotSize, height: self.dotSize)
                .offset(x: CGFloat(dot.position.x) - 350, y: CGFloat(dot.position.y) - 350)
        }
    }
    
    func dotColor(dot: Dot) -> Color {
        if dot.success {
            return .green
        }
        
        if dot.dead {
            return .red
        }
        
        if dot.champion {
            return .yellow
        }
        
        return .white
    }
}

struct ContentView: View {
    let timer = Timer.publish(
        every: (1.0 / 60.0),
        on: .main,
        in: .common
    ).autoconnect()
    
    @ObservedObject var population = Population(
        populationSize: 80,
        width: 700,
        height: 700,
        dotSize: 10,
        brainSize: 300,
        minTargetDistance: 5,
        mutationRatio: 0.01
    )
    
    let target = Vector(x: 50, y: 50)
    
    var body: some View {
        return ZStack {
            Color.indigo.edgesIgnoringSafeArea(.all)
            Rectangle()
                .stroke(lineWidth: CGFloat(5.0))
                .foregroundColor(.white)
                .frame(width: CGFloat(700), height: CGFloat(700))
            if(population.generation > 1) {
                Text("Generation: \(population.generation)\nNumber of steps: \(population.minStep)")
            }
            else {
                Text("Generation: \(population.generation)")
            }
            Rectangle()
                .foregroundColor(.black)
                .frame(width: 20, height: 20)
                .offset(x: CGFloat(target.x) - 350, y: CGFloat(target.y) - 350)
            ForEach (population.dots) { (dot: Dot) in
                DotView(dot, 10)
            }
            DotView(population.dots[0], 15)
        }.onReceive(timer) { input in
            if self.population.allDead() {
                self.population.naturalSelection(target: self.target)
            } else {
                self.population.update(target: self.target)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
