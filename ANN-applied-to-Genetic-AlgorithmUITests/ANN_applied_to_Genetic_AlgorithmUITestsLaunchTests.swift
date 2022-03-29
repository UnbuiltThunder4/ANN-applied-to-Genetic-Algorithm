//
//  ANN_applied_to_Genetic_AlgorithmUITestsLaunchTests.swift
//  ANN-applied-to-Genetic-AlgorithmUITests
//
//  Created by Eugenio Raja on 26/03/22.
//

import XCTest

class ANN_applied_to_Genetic_AlgorithmUITestsLaunchTests: XCTestCase {

    override class var runsForEachTargetApplicationUIConfiguration: Bool {
        true
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testLaunch() throws {
        let app = XCUIApplication()
        app.launch()

        // Insert steps here to perform after app launch but before taking a screenshot,
        // such as logging into a test account or navigating somewhere in the app

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Launch Screen"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
