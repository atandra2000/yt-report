// OCR helper for macOS Vision. One process, many images.
//
// Why a subprocess at all: Vision lives in Objective-C/Swift, and the rest of this
// pipeline is Python. A per-image process would pay interpreter start-up and
// framework load for every frame; instead the Python side sends a JSON job on
// stdin and gets a JSON result on stdout, so the whole run is one process.
//
// Bounding boxes come back in PIXELS of the source image, not Vision's normalised
// lower-left-origin units: provenance that cites a box is checked against the
// frame's own pixel dimensions later, and a caller that has to re-derive the
// transform to do that will eventually get it wrong.
//
// The original image is read, never a re-encoded or upscaled copy: the pixels
// that get read are the pixels that were extracted, which is what makes a
// citation to a box mean something.
import Foundation
import Vision
import AppKit

struct Job: Decodable {
    var images: [String]
    var languages: [String]?
    var min_height: Double?
    var language_correction: Bool?
}

struct Box: Encodable {
    var text: String
    var confidence: Double
    // [x, y, w, h] in source pixels, top-left origin (flipped from Vision's
    // bottom-left so it matches how anyone reads an image).
    var bbox: [Double]
}

struct ImageResult: Encodable {
    var status: String
    var width: Int
    var height: Int
    var boxes: [Box]
    var error: String?
}

struct Response: Encodable {
    var backend: String
    var version: String
    var results: [String: ImageResult]
}

func readImage(_ path: String) -> (NSImage, Int, Int)? {
    guard let img = NSImage(contentsOfFile: path) else { return nil }
    var rect = CGRect(origin: .zero, size: img.size)
    guard let cg = img.cgImage(forProposedRect: &rect, context: nil, hints: nil) else { return nil }
    return (img, cg.width, cg.height)
}

let raw = FileHandle.standardInput.readDataToEndOfFile()
guard let job = try? JSONDecoder().decode(Job.self, from: raw) else {
    FileHandle.standardError.write("ocr_vision: unreadable job\n".data(using: .utf8)!)
    exit(2)
}

var results: [String: ImageResult] = [:]

for path in job.images {
    guard let (_, w, h) = readImage(path) else {
        results[path] = ImageResult(status: "unreadable", width: 0, height: 0, boxes: [], error: "cannot open image")
        continue
    }
    guard let image = NSImage(contentsOfFile: path),
          let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        results[path] = ImageResult(status: "unreadable", width: w, height: h, boxes: [], error: "cannot decode image")
        continue
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = job.language_correction ?? true
    // No automatic language detection: silently reading slide text in whatever
    // language Vision guesses is how a price comes back as a word. The caller
    // names the languages or gets nothing.
    if let langs = job.languages, !langs.isEmpty {
        request.recognitionLanguages = langs
    }
    if let mh = job.min_height {
        request.minimumTextHeight = Float(mh)
    }

    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    var boxes: [Box] = []
    var status = "ok"
    do {
        try handler.perform([request])
        for obs in (request.results ?? []) {
            guard let top = obs.topCandidates(1).first else { continue }
            let bb = obs.boundingBox
            boxes.append(Box(
                text: top.string,
                confidence: Double(top.confidence),
                bbox: [bb.origin.x * Double(w),
                       (1.0 - bb.origin.y - bb.height) * Double(h),
                       bb.width * Double(w),
                       bb.height * Double(h)]
            ))
        }
    } catch {
        status = "failed"
        results[path] = ImageResult(status: status, width: w, height: h, boxes: [], error: "\(error)")
        continue
    }
    results[path] = ImageResult(status: boxes.isEmpty ? "no_text" : "ok", width: w, height: h, boxes: boxes)
}

let version = (Bundle.main.infoDictionary?["CFBundleVersion"] as? String) ?? "unknown"
let response = Response(backend: "apple-vision", version: "macos-vision-1", results: results)
let out = try JSONEncoder().encode(response)
FileHandle.standardOutput.write(out)
