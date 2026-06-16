
import SwiftUI
import CoreMotion
import Charts

// MARK: - Replace with your Railway deployment URL before running
private let baseURL = "https://your-app.railway.app"

// MARK: - Data Models

struct SessionSummary: Codable, Identifiable {
    let id: String
    let startedAt: String
    let durationS: Double
    let sampleCount: Int
    let analysis: AnalysisResult?

    enum CodingKeys: String, CodingKey {
        case id
        case startedAt = "started_at"
        case durationS = "duration_s"
        case sampleCount = "sample_count"
        case analysis
    }
}

struct AnalysisResult: Codable {
    let betaX: Double
    let betaY: Double
    let betaZ: Double
    let betaMagnitude: Double
    let r2X: Double
    let r2Y: Double
    let r2Z: Double
    let r2Magnitude: Double

    enum CodingKeys: String, CodingKey {
        case betaX = "beta_x"
        case betaY = "beta_y"
        case betaZ = "beta_z"
        case betaMagnitude = "beta_magnitude"
        case r2X = "r2_x"
        case r2Y = "r2_y"
        case r2Z = "r2_z"
        case r2Magnitude = "r2_magnitude"
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
        writeSampleToCSV(t: elapsed, x: x, y: y, z: z, magnitude: mag)
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
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds, .withDashSeparatorInDate, .withColonSeparatorInTime]
        guard let date = f.date(from: session.startedAt) else { return session.startedAt }
        let d = DateFormatter(); d.dateStyle = .medium; d.timeStyle = .short
        return d.string(from: date)
    }

    private var betaColor: Color {
        guard let a = session.analysis else { return .gray }
        if a.betaMagnitude >= 0.8 && a.betaMagnitude <= 1.5 { return .green }
        if a.betaMagnitude > 1.5 && a.betaMagnitude <= 2.5 { return .yellow }
        return .red
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(formattedDate).font(.subheadline.bold())
            HStack {
                Text(String(format: "%.1f s  •  %d samples", session.durationS, session.sampleCount))
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let a = session.analysis {
                    Label(String(format: "β=%.2f", a.betaMagnitude), systemImage: "chart.line.downtrend.xyaxis")
                        .font(.caption.bold()).foregroundStyle(betaColor)
                } else {
                    Text("No analysis").font(.caption).foregroundStyle(.tertiary)
                }
            }
        }.padding(.vertical, 2)
    }
}

// MARK: - SessionDetailView

struct BetaBarEntry: Identifiable {
    let id = UUID()
    let axis: String
    let beta: Double
    let r2: Double
}

struct SessionDetailView: View {
    let session: SessionSummary

    private var barEntries: [BetaBarEntry] {
        guard let a = session.analysis else { return [] }
        return [
            BetaBarEntry(axis: "X",   beta: a.betaX,         r2: a.r2X),
            BetaBarEntry(axis: "Y",   beta: a.betaY,         r2: a.r2Y),
            BetaBarEntry(axis: "Z",   beta: a.betaZ,         r2: a.r2Z),
            BetaBarEntry(axis: "Mag", beta: a.betaMagnitude, r2: a.r2Magnitude),
        ]
    }

    private var formattedDate: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds, .withDashSeparatorInDate, .withColonSeparatorInTime]
        guard let date = f.date(from: session.startedAt) else { return session.startedAt }
        let d = DateFormatter(); d.dateStyle = .long; d.timeStyle = .medium
        return d.string(from: date)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(formattedDate).font(.subheadline).foregroundStyle(.secondary)
                    Text(String(format: "Duration: %.1f s  •  %d samples", session.durationS, session.sampleCount))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Divider()

                if !barEntries.isEmpty {
                    Text("Power Law Exponent (β)").font(.headline)

                    Chart {
                        ForEach(barEntries) { entry in
                            BarMark(x: .value("Axis", entry.axis), y: .value("Beta", entry.beta))
                                .foregroundStyle(barColor(for: entry.beta))
                                .annotation(position: .top) {
                                    Text(String(format: "%.2f", entry.beta))
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                        }
                        RuleMark(y: .value("Healthy baseline", 1.0))
                            .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                            .foregroundStyle(.gray.opacity(0.7))
                            .annotation(position: .trailing, alignment: .leading) {
                                Text("β=1 baseline").font(.caption2).foregroundStyle(.gray)
                            }
                    }
                    .frame(height: 240)
                    .chartYAxisLabel("β exponent")
                    .chartXAxisLabel("Axis")

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Goodness of Fit (R²)").font(.subheadline.bold())
                        ForEach(barEntries) { entry in
                            HStack {
                                Text(entry.axis).frame(width: 40, alignment: .leading).font(.caption.bold())
                                ProgressView(value: min(max(entry.r2, 0), 1))
                                    .tint(entry.r2 >= 0.7 ? .green : (entry.r2 >= 0.4 ? .yellow : .red))
                                Text(String(format: "%.3f", entry.r2))
                                    .font(.caption.monospacedDigit()).frame(width: 50, alignment: .trailing)
                            }
                        }
                    }
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Interpretation").font(.subheadline.bold())
                        Text("β ≈ 0    White noise (uncorrelated)")
                        Text("β ≈ 1    1/f pink noise — typical healthy tremor")
                        Text("β ≈ 2    Brownian / random-walk motion")
                        Text("β > 2    Highly correlated, low-freq tremor dominant")
                    }
                    .font(.caption).foregroundStyle(.secondary).padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))

                } else {
                    ContentUnavailableView("No Analysis Available", systemImage: "chart.bar.xaxis",
                        description: Text("Too few samples to compute power law fit."))
                }
            }.padding()
        }
        .navigationTitle("Session Detail")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func barColor(for beta: Double) -> Color {
        if beta >= 0.8 && beta <= 1.5 { return .green }
        if beta > 1.5 && beta <= 2.5 { return .yellow }
        return .red
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
