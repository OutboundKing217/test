import SwiftUI
import CoreMotion
import Charts

private let baseURL = "https://web-production-8cb5b.up.railway.app"

// MARK: - Data Models

struct SessionSummary: Codable, Identifiable {
    let id: String
    let startedAt: String
    let durationS: Double
    let sampleCount: Int
    let analysis: AnalysisResult?

    enum CodingKeys: String, CodingKey {
        case id = "session_id"
        case startedAt = "started_at"
        case durationS = "duration_s"
        case sampleCount = "sample_count"
        case analysis
    }
}

struct AnalysisResult: Codable {
    let tau: Double?
    let powerLawRange: Double?
    let goodnessOfFit: Double?
    let isScaleFree: Bool?
    let nEvents: Int?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case tau
        case powerLawRange = "power_law_range"
        case goodnessOfFit = "goodness_of_fit"
        case isScaleFree = "is_scale_free"
        case nEvents = "n_events"
        case error
    }
}

struct SampleEntry: Codable {
    let t: Double
    let x: Double
    let y: Double
    let z: Double
    let magnitude: Double
}

// MARK: - AccelerometerManager

@Observable
class AccelerometerManager {
    let userID: String

    var isRecording = false
    var elapsedTime: Double = 0
    var currentX: Double = 0
    var currentY: Double = 0
    var currentZ: Double = 0
    var currentMagnitude: Double = 0

    var sessions: [SessionSummary] = []
    var isUploading = false
    var uploadError: String?

    private let motionManager = CMMotionManager()
    private let updateInterval: TimeInterval = 1.0 / 30.0

    private var csvFileURL: URL?
    private var fileHandle: FileHandle?
    private var recordingStartDate: Date?
    private var startTime: TimeInterval?
    private var buffer: [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)] = []

    init() {
        if let existing = UserDefaults.standard.string(forKey: "userID"), !existing.isEmpty {
            userID = existing
        } else {
            let newID = UUID().uuidString
            UserDefaults.standard.set(newID, forKey: "userID")
            userID = newID
        }
    }

    func startUpdates() {
        guard motionManager.isAccelerometerAvailable else { return }
        guard !isRecording else { return }

        buffer = []
        uploadError = nil
        recordingStartDate = Date()
        startTime = nil
        elapsedTime = 0

        setupCSVFile()

        motionManager.accelerometerUpdateInterval = updateInterval
        motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
            guard let self, let data, error == nil else { return }
            self.handleSample(data)
        }

        isRecording = true
    }

    func stopUpdates() {
        guard isRecording else { return }
        motionManager.stopAccelerometerUpdates()
        isRecording = false
        closeCSVFile()
        Task { await self.uploadSession() }
    }

    private func handleSample(_ data: CMAccelerometerData) {
        let now = data.timestamp
        if startTime == nil { startTime = now }
        let elapsed = now - (startTime ?? now)

        let x = data.acceleration.x
        let y = data.acceleration.y
        let z = data.acceleration.z
        let mag = sqrt(x * x + y * y + z * z)

        currentX = x; currentY = y; currentZ = z
        currentMagnitude = mag; elapsedTime = elapsed

        buffer.append((t: elapsed, x: x, y: y, z: z, magnitude: mag))
        writeSampleToCSV(t: elapsed, x: x, y: y, z: y, magnitude: mag)
    }

    private func setupCSVFile() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let url = docs.appendingPathComponent("accel_\(formatter.string(from: Date())).csv")
        csvFileURL = url
        try? "elapsedSeconds,x,y,z,magnitude\n".write(to: url, atomically: true, encoding: .utf8)
        fileHandle = try? FileHandle(forWritingTo: url)
        fileHandle?.seekToEndOfFile()
    }

    private func writeSampleToCSV(t: Double, x: Double, y: Double, z: Double, magnitude: Double) {
        if let data = "\(t),\(x),\(y),\(z),\(magnitude)\n".data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    private func closeCSVFile() {
        fileHandle?.closeFile()
        fileHandle = nil
    }

    func uploadSession() async {
        guard !buffer.isEmpty else { return }
        await MainActor.run { isUploading = true; uploadError = nil }

        let samples = buffer.map { ["t": $0.t, "x": $0.x, "y": $0.y, "z": $0.z, "magnitude": $0.magnitude] }
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let body: [String: Any] = [
            "user_id": userID,
            "started_at": isoFormatter.string(from: recordingStartDate ?? Date()),
            "samples": samples
        ]

        guard let url = URL(string: "\(baseURL)/sessions") else {
            await MainActor.run { isUploading = false; uploadError = "Invalid server URL" }
            return
        }

        do {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (_, response) = try await URLSession.shared.data(for: request)
            if let r = response as? HTTPURLResponse, r.statusCode >= 300 { throw URLError(.badServerResponse) }
            await MainActor.run { isUploading = false }
            await fetchSessions()
        } catch {
            await MainActor.run { isUploading = false; uploadError = "Upload failed: \(error.localizedDescription)" }
        }
    }

    func fetchSessions() async {
        guard let url = URL(string: "\(baseURL)/users/\(userID)/sessions") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let decoded = try JSONDecoder().decode([SessionSummary].self, from: data)
            await MainActor.run { sessions = decoded }
        } catch {
            print("fetchSessions error: \(error)")
        }
    }
}

// MARK: - AccelerometerView

struct AccelerometerView: View {
    @Bindable var manager: AccelerometerManager

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                VStack(spacing: 8) {
                    Text("Live Accelerometer").font(.headline)
                    HStack(spacing: 20) {
                        axisCard("X", value: manager.currentX, color: .red)
                        axisCard("Y", value: manager.currentY, color: .green)
                        axisCard("Z", value: manager.currentZ, color: .blue)
                    }
                    HStack {
                        Image(systemName: "waveform.path")
                        Text(String(format: "Magnitude: %.4f g", manager.currentMagnitude)).monospacedDigit()
                    }.foregroundStyle(.purple)
                    Text(String(format: "Elapsed: %.1f s", manager.elapsedTime))
                        .foregroundStyle(.secondary).monospacedDigit()
                }
                .padding()
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

                if manager.isUploading {
                    HStack { ProgressView(); Text("Uploading session...").foregroundStyle(.secondary) }
                } else if let err = manager.uploadError {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        Text(err).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                    }.padding(.horizontal)
                }

                Button(action: { manager.isRecording ? manager.stopUpdates() : manager.startUpdates() }) {
                    Label(
                        manager.isRecording ? "Stop Recording" : "Start Recording",
                        systemImage: manager.isRecording ? "stop.circle.fill" : "record.circle"
                    )
                    .font(.title3.bold()).foregroundStyle(.white).padding()
                    .frame(maxWidth: .infinity)
                    .background(manager.isRecording ? Color.red : Color.accentColor,
                                in: RoundedRectangle(cornerRadius: 14))
                }.padding(.horizontal)

                Spacer()
            }
            .padding()
            .navigationTitle("NeuroMotion")
        }
    }

    private func axisCard(_ label: String, value: Double, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label).font(.caption.bold()).foregroundStyle(color)
            Text(String(format: "%.4f", value)).font(.caption2).monospacedDigit()
        }
        .frame(minWidth: 60).padding(8)
        .background(color.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - SessionHistoryView

struct SessionHistoryView: View {
    @Bindable var manager: AccelerometerManager

    var body: some View {
        NavigationStack {
            Group {
                if manager.sessions.isEmpty {
                    ContentUnavailableView("No Sessions Yet", systemImage: "waveform",
                        description: Text("Record and upload your first session to see history."))
                } else {
                    List(manager.sessions) { session in
                        NavigationLink(destination: SessionDetailView(session: session)) {
                            SessionRowView(session: session)
                        }
                    }
                }
            }
            .navigationTitle("History")
            .refreshable { await manager.fetchSessions() }
        }
    }
}

// MARK: - SessionRowView

struct SessionRowView: View {
    let session: SessionSummary

    private var formattedDate: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds,
                           .withDashSeparatorInDate, .withColonSeparatorInTime]
        guard let date = f.date(from: session.startedAt) else { return session.startedAt }
        let d = DateFormatter(); d.dateStyle = .medium; d.timeStyle = .short
        return d.string(from: date)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(formattedDate).font(.subheadline.bold())
            HStack {
                Text(String(format: "%.1f s  •  %d samples", session.durationS, session.sampleCount))
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let a = session.analysis {
                    if let tau = a.tau {
                        Label(String(format: "τ=%.2f", tau), systemImage: "chart.line.downtrend.xyaxis")
                            .font(.caption.bold())
                            .foregroundStyle(a.isScaleFree == true ? .green : .orange)
                    } else {
                        Text(a.error != nil ? "No fit" : "Pending")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                } else {
                    Text("No analysis").font(.caption).foregroundStyle(.tertiary)
                }
            }
        }.padding(.vertical, 2)
    }
}

// MARK: - SessionDetailView

struct SessionDetailView: View {
    let session: SessionSummary

    private var formattedDate: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds,
                           .withDashSeparatorInDate, .withColonSeparatorInTime]
        guard let date = f.date(from: session.startedAt) else { return session.startedAt }
        let d = DateFormatter(); d.dateStyle = .long; d.timeStyle = .medium
        return d.string(from: date)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {

                // Header
                VStack(alignment: .leading, spacing: 6) {
                    Text(formattedDate).font(.subheadline).foregroundStyle(.secondary)
                    Text(String(format: "Duration: %.1f s  •  %d samples",
                                session.durationS, session.sampleCount))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Divider()

                if let a = session.analysis, a.tau != nil {
                    // Scale-free badge
                    HStack {
                        Image(systemName: a.isScaleFree == true ? "checkmark.seal.fill" : "xmark.seal.fill")
                            .foregroundStyle(a.isScaleFree == true ? .green : .red)
                            .font(.title2)
                        Text(a.isScaleFree == true ? "Scale-Free Dynamics" : "Not Scale-Free")
                            .font(.title3.bold())
                            .foregroundStyle(a.isScaleFree == true ? .green : .red)
                        Spacer()
                    }
                    .padding()
                    .background((a.isScaleFree == true ? Color.green : Color.red).opacity(0.1),
                                in: RoundedRectangle(cornerRadius: 12))

                    // Tau
                    VStack(spacing: 12) {
                        VStack(spacing: 4) {
                            Text("Power Law Exponent").font(.subheadline).foregroundStyle(.secondary)
                            Text(String(format: "τ = %.3f", a.tau!))
                                .font(.system(size: 48, weight: .bold, design: .rounded))
                                .foregroundStyle(.primary)
                            Text("Ideal range: 1.5 – 2.5").font(.caption).foregroundStyle(.secondary)
                        }

                        Divider()

                        // Goodness of fit
                        if let gof = a.goodnessOfFit {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("Goodness of Fit").font(.subheadline.bold())
                                    Spacer()
                                    Text(String(format: "%.1f%%", gof * 100))
                                        .font(.subheadline.bold())
                                        .foregroundStyle(gof >= 0.8 ? .green : (gof >= 0.6 ? .yellow : .red))
                                }
                                ProgressView(value: gof)
                                    .tint(gof >= 0.8 ? .green : (gof >= 0.6 ? .yellow : .red))
                                Text("≥ 80% required for scale-free classification")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }

                        Divider()

                        // Power law range + events
                        HStack(spacing: 20) {
                            if let plr = a.powerLawRange {
                                statBox(label: "Fit Range", value: String(format: "%.2f dec", plr))
                            }
                            if let n = a.nEvents {
                                statBox(label: "Events", value: "\(n)")
                            }
                        }
                    }
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

                    // Interpretation
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Interpretation").font(.subheadline.bold())
                        Text("τ ≈ 1.5    Branching ratio near critical point")
                        Text("τ ≈ 2.0    Mean-field critical exponent")
                        Text("τ < 1.5    Sub-critical, reduced complexity")
                        Text("τ > 2.5    Super-critical, excessive synchrony")
                        Text("")
                        Text("Scale-free movement dynamics (goodness of fit ≥ 80%) indicate healthy neurological function. Loss of scale-free behavior may reflect disease progression.")
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(.caption).foregroundStyle(.secondary)
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))

                } else {
                    // No analysis or error
                    VStack(spacing: 8) {
                        Image(systemName: "chart.bar.xaxis").font(.largeTitle).foregroundStyle(.secondary)
                        Text("No Analysis Available").font(.headline)
                        if let a = session.analysis, let err = a.error {
                            Text(err).font(.caption).foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                        } else {
                            Text("Too few samples or events detected.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity).padding(40)
                }
            }.padding()
        }
        .navigationTitle("Session Detail")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func statBox(label: String, value: String) -> some View {
        VStack(spacing: 4) {
            Text(value).font(.title3.bold())
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(10)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - ContentView

struct ContentView: View {
    @State private var manager = AccelerometerManager()

    var body: some View {
        TabView {
            AccelerometerView(manager: manager)
                .tabItem { Label("Record", systemImage: "waveform.circle") }

            SessionHistoryView(manager: manager)
                .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
                .onAppear { Task { await manager.fetchSessions() } }
        }
    }
}

@main
struct AccelerometerApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
