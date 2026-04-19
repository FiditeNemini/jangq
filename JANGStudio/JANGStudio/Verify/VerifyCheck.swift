// JANGStudio/JANGStudio/Verify/VerifyCheck.swift
import Foundation

enum VerifyID: String, CaseIterable {
    case jangConfigExists, jangConfigFormat, schemaValid, capabilitiesPresent,
         chatTemplate, tokenizerFiles, shardsMatchIndex, vlPreprocessors,
         videoPreprocessors, generationConfig, layerCountSane,
         miniMaxCustomPy, tokenizerClassConcrete
}

struct VerifyCheck: Identifiable, Equatable {
    let id: VerifyID
    let title: String
    let status: PreflightStatus
    let required: Bool
    let hint: String?
}
