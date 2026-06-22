import SwiftUI
import CoreMotion
import Charts
import WatchConnectivity

private let sensorRecorderDuration: TimeInterval = 12 * 60 * 60
private let recordedSessionStartKey = "recordedSessionStart"
private let recordedSessionEndKey = "recordedSessionEnd"

// MARK: - Data Models

struct LocalSession: Identifiable {
    let id = UUID()
    let fileURL: URL
    let name: String
    let date: Date
    let sampleCount: Int

    var formattedDate: String {
        let d = DateFormatter()
        d.dateStyle = .medium
        d.timeStyle = .short
        return d.string(from: date)
    }

    var formattedSize: String {
        let bytes = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let mb = Double(bytes) / 1_000_000
        return mb >= 1 ? String(format: "%.1f MB", mb) : String(format: "%.0f KB", Double(bytes) / 1000)
    }
}

struct SampleEntry: Codable {
    let t: Double
    let x: Double
    let y: Double
    let z: Double
    let magnitude: Double
}

struct WatchSessionTransferPayload: Decodable {
    let startedAt: String
    let samples: [SampleEntry]

    enum CodingKeys: String, CodingKey {
        case startedAt = "started_at"
        case samples
    }
}

struct LivePoint: Identifiable {
    let id = UUID()
    let t: Double
    let value: Double
}

// MARK: - AccelerometerManager

@Observable
class AccelerometerManager: NSObject, WCSessionDelegate {
    var userName: String

    var isRecording = false
    var elapsedTime: Double = 0
    var currentX: Double = 0
    var currentY: Double = 0
    var currentZ: Double = 0
    var currentMagnitude: Double = 0

    var liveChartPoints: [LivePoint] = []
    private let maxChartPoints = 150

    var localSessions: [LocalSession] = []
    var statusMessage: String?

    private let motionManager = CMMotionManager()
    private let sensorRecorder = CMSensorRecorder()
    private let updateInterval: TimeInterval = 1.0 / 30.0

    private var csvFileURL: URL?
    private var fileHandle: FileHandle?
    private var recordingStartDate: Date?
    private var startTime: TimeInterval?

    private var gravX: Double = 0
    private var gravY: Double = 0
    private var gravZ: Double = 0
    private let gravAlpha: Double = 0.95

    private let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    override init() {
        userName = UserDefaults.standard.string(forKey: "userName") ?? ""
        super.init()
        configureWatchConnectivity()
        loadLocalSessions()
        restorePendingRecording()
        Task { await recoverFinishedRecordingIfAvailable() }
    }

    func saveName(_ name: String) {
        userName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        UserDefaults.standard.set(userName, forKey: "userName")
    }

    private func configureWatchConnectivity() {
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
    }

    // MARK: - Local File Management

    private var documentsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    func loadLocalSessions() {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: documentsDir, includingPropertiesForKeys: [.creationDateKey, .fileSizeKey]) else { return }
        let csvFiles = files.filter { $0.pathExtension == "csv" }.sorted {
            let d1 = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let d2 = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return d1 > d2
        }
        localSessions = csvFiles.map { url in
            let date = (try? url.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date()
            let lines = (try? String(contentsOf: url, encoding: .utf8).components(separatedBy: "\n").count - 2) ?? 0
            return LocalSession(fileURL: url, name: url.deletingPathExtension().lastPathComponent, date: date, sampleCount: max(0, lines))
        }
    }

    func deleteSession(_ session: LocalSession) {
        try? FileManager.default.removeItem(at: session.fileURL)
        loadLocalSessions()
    }

    // MARK: - Recording

    func startUpdates() {
        guard motionManager.isAccelerometerAvailable, !isRecording else { return }
        liveChartPoints = []
        statusMessage = nil
        recordingStartDate = Date()
        startTime = nil
        elapsedTime = 0
        gravX = 0; gravY = 0; gravZ = 0
        setupCSVFile()
        startSensorRecording(at: recordingStartDate!)
        startLiveUpdates()
        isRecording = true
    }

    func stopUpdates() {
        guard isRecording else { return }
        motionManager.stopAccelerometerUpdates()
        isRecording = false
        closeCSVFile()
        finishSensorRecording(at: Date())
        Task { await self.processAfterStop() }
    }

    private func restorePendingRecording() {
        guard let startDate = UserDefaults.standard.object(forKey: recordedSessionStartKey) as? Date,
              let endDate = UserDefaults.standard.object(forKey: recordedSessionEndKey) as? Date else { return }
        recordingStartDate = startDate
        if endDate > Date() {
            isRecording = true
            elapsedTime = Date().timeIntervalSince(startDate)
            statusMessage = "Background recording active."
            startLiveUpdates()
        }
    }

    private func startLiveUpdates() {
        guard motionManager.isAccelerometerAvailable else { return }
        motionManager.accelerometerUpdateInterval = updateInterval
        motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
            guard let self, let data, error == nil else { return }
            self.handleSample(data)
        }
    }

    private func startSensorRecording(at startDate: Date) {
        guard CMSensorRecorder.isAccelerometerRecordingAvailable() else {
            statusMessage = "Background recording not available on this device."
            return
        }
        UserDefaults.standard.set(startDate, forKey: recordedSessionStartKey)
        UserDefaults.standard.set(startDate.addingTimeInterval(sensorRecorderDuration), forKey: recordedSessionEndKey)
        sensorRecorder.recordAccelerometer(forDuration: sensorRecorderDuration)
        statusMessage = "Background recording enabled."
    }

    private func finishSensorRecording(at endDate: Date) {
        guard UserDefaults.standard.object(forKey: recordedSessionStartKey) != nil else { return }
        UserDefaults.standard.set(endDate, forKey: recordedSessionEndKey)
    }

    private func handleSample(_ data: CMAccelerometerData) {
        let now = data.timestamp
        if startTime == nil { startTime = now }

        let x = data.acceleration.x
        let y = data.acceleration.y
        let z = data.acceleration.z
        let mag = sqrt(x*x + y*y + z*z)

        currentX = x; currentY = y; currentZ = z
        currentMagnitude = mag
        elapsedTime = recordingStartDate.map { Date().timeIntervalSince($0) } ?? (now - (startTime ?? now))

        if gravX == 0 && gravY == 0 && gravZ == 0 {
            gravX = x; gravY = y; gravZ = z
        } else {
            gravX = gravAlpha * gravX + (1 - gravAlpha) * x
            gravY = gravAlpha * gravY + (1 - gravAlpha) * y
            gravZ = gravAlpha * gravZ + (1 - gravAlpha) * z
        }
        let dynMag = sqrt(pow(x - gravX, 2) + pow(y - gravY, 2) + pow(z - gravZ, 2))

        liveChartPoints.append(LivePoint(t: elapsedTime, value: dynMag))
        if liveChartPoints.count > maxChartPoints { liveChartPoints.removeFirst() }

        writeSampleToCSV(t: elapsedTime, x: x, y: y, z: z, magnitude: mag)
    }

    // MARK: - CSV File

    private func setupCSVFile() {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let filename = "\(userName.isEmpty ? "recording" : userName)_\(formatter.string(from: Date())).csv"
        let url = documentsDir.appendingPathComponent(filename)
        csvFileURL = url
        try? "elapsedSeconds,x,y,z,magnitude\n".write(to: url, atomically: true, encoding: .utf8)
        fileHandle = try? FileHandle(forWritingTo: url)
        fileHandle?.seekToEndOfFile()
    }

    private func writeSampleToCSV(t: Double, x: Double, y: Double, z: Double, magnitude: Double) {
        let line = "\(String(format: "%.4f", t)),\(String(format: "%.6f", x)),\(String(format: "%.6f", y)),\(String(format: "%.6f", z)),\(String(format: "%.6f", magnitude))\n"
        if let data = line.data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    private func closeCSVFile() {
        fileHandle?.closeFile()
        fileHandle = nil
    }

    // MARK: - Post-recording

    private func processAfterStop() async {
        let didWriteFromRecorder = await writeFromSensorRecorderToCSV()
        if !didWriteFromRecorder {
            await MainActor.run {
                statusMessage = "Recording saved."
                loadLocalSessions()
            }
        }
        clearPendingRecordedSession()
    }

    private func recoverFinishedRecordingIfAvailable() async {
        guard let endDate = UserDefaults.standard.object(forKey: recordedSessionEndKey) as? Date,
              endDate <= Date() else { return }
        await MainActor.run {
            isRecording = false
            motionManager.stopAccelerometerUpdates()
            closeCSVFile()
        }
        let success = await writeFromSensorRecorderToCSV()
        if !success {
            await MainActor.run { statusMessage = "Recorded data not ready yet. Reopen app in a few minutes." }
        }
        clearPendingRecordedSession()
    }

    private func writeFromSensorRecorderToCSV() async -> Bool {
        guard let startedAt = UserDefaults.standard.object(forKey: recordedSessionStartKey) as? Date,
              let requestedEnd = UserDefaults.standard.object(forKey: recordedSessionEndKey) as? Date else {
            return false
        }
        let endedAt = min(requestedEnd, Date())
        guard endedAt > startedAt else { return false }

        let windowDuration: TimeInterval = 1800
        var windowStart = startedAt
        var chunkIndex = 0
        var successCount = 0
        let totalChunks = max(1, Int(ceil(endedAt.timeIntervalSince(startedAt) / windowDuration)))

        while windowStart < endedAt {
            let windowEnd = min(windowStart.addingTimeInterval(windowDuration), endedAt)
            chunkIndex += 1

            await MainActor.run {
                statusMessage = "Saving chunk \(chunkIndex) of \(totalChunks)..."
            }

            if let samples = recordedSamples(from: windowStart, to: windowEnd), !samples.isEmpty {
                let formatter = DateFormatter()
                formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
                let filename = "\(userName.isEmpty ? "recording" : userName)_\(formatter.string(from: windowStart))_chunk\(chunkIndex).csv"
                let url = documentsDir.appendingPathComponent(filename)
                var csv = "elapsedSeconds,x,y,z,magnitude\n"
                for s in samples {
                    csv += "\(String(format: "%.4f", s.t)),\(String(format: "%.6f", s.x)),\(String(format: "%.6f", s.y)),\(String(format: "%.6f", s.z)),\(String(format: "%.6f", s.magnitude))\n"
                }
                try? csv.write(to: url, atomically: true, encoding: .utf8)
                successCount += 1
            }

            windowStart = windowEnd
            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        await MainActor.run {
            statusMessage = successCount > 0 ? "Saved \(successCount) chunk\(successCount == 1 ? "" : "s") to Files." : nil
            loadLocalSessions()
        }

        return successCount > 0
    }

    private func recordedSamples(from startedAt: Date, to endedAt: Date) -> [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)]? {
        guard let dataList = sensorRecorder.accelerometerData(from: startedAt, to: endedAt) else { return nil }
        var samples: [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)] = []
        var iterator = NSFastEnumerationIterator(dataList)
        while let data = iterator.next() as? CMRecordedAccelerometerData {
            let x = data.acceleration.x
            let y = data.acceleration.y
            let z = data.acceleration.z
            let elapsed = data.startDate.timeIntervalSince(startedAt)
            samples.append((t: elapsed, x: x, y: y, z: z, magnitude: sqrt(x*x + y*y + z*z)))
        }
        return samples.isEmpty ? nil : samples
    }

    private func clearPendingRecordedSession() {
        UserDefaults.standard.removeObject(forKey: recordedSessionStartKey)
        UserDefaults.standard.removeObject(forKey: recordedSessionEndKey)
    }

    // MARK: - Watch Data

    private func receiveWatchSessionData(_ data: Data) async {
        do {
            let payload = try JSONDecoder().decode(WatchSessionTransferPayload.self, from: data)
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            let startDate = isoFormatter.date(from: payload.startedAt) ?? Date()
            let filename = "watch_\(userName.isEmpty ? "recording" : userName)_\(formatter.string(from: startDate)).csv"
            let url = documentsDir.appendingPathComponent(filename)
            var csv = "elapsedSeconds,x,y,z,magnitude\n"
            for s in payload.samples {
                csv += "\(String(format: "%.4f", s.t)),\(String(format: "%.6f", s.x)),\(String(format: "%.6f", s.y)),\(String(format: "%.6f", s.z)),\(String(format: "%.6f", s.magnitude))\n"
            }
            try csv.write(to: url, atomically: true, encoding: .utf8)
            await MainActor.run {
                statusMessage = "Watch data saved: \(payload.samples.count) samples."
                loadLocalSessions()
            }
        } catch {
            await MainActor.run { statusMessage = "Could not save watch data: \(error.localizedDescription)" }
        }
    }
}

// MARK: - WCSessionDelegate

extension AccelerometerManager {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {}

    func session(_ session: WCSession, didReceiveMessage message: [String: Any], replyHandler: @escaping ([String: Any]) -> Void) {
        replyHandler([:])
    }

    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        guard file.metadata?["kind"] as? String == "watchSessionUpload" else { return }
        do {
            let data = try Data(contentsOf: file.fileURL)
            Task { await receiveWatchSessionData(data) }
        } catch {
            Task { @MainActor in statusMessage = "Could not read watch file: \(error.localizedDescription)" }
        }
    }

    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) { session.activate() }
}

// MARK: - OnboardingView

struct OnboardingView: View {
    @Bindable var manager: AccelerometerManager
    @State private var nameInput = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            Spacer()
            VStack(spacing: 16) {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 72))
                    .foregroundStyle(.blue)
                Text("NeuroMotion")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                Text("Movement tracking for neurological health")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Spacer()
            VStack(alignment: .leading, spacing: 8) {
                Text("Your Name")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 4)
                TextField("Enter your full name", text: $nameInput)
                    .font(.body)
                    .padding()
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                    .focused($focused)
                    .submitLabel(.go)
                    .onSubmit { confirmName() }
            }
            .padding(.horizontal)
            Button(action: confirmName) {
                Text("Get Started")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        nameInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray : Color.blue,
                        in: RoundedRectangle(cornerRadius: 14)
                    )
            }
            .disabled(nameInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .padding(.horizontal)
            .padding(.top, 12)
            Spacer().frame(height: 48)
        }
        .onAppear { focused = true }
    }

    private func confirmName() {
        let trimmed = nameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        manager.saveName(trimmed)
    }
}

// MARK: - LiveChartView

struct LiveChartView: View {
    let points: [LivePoint]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Circle().fill(Color.green).frame(width: 7, height: 7)
                Text("Dynamic Magnitude").font(.caption.bold()).foregroundStyle(.secondary)
            }
            if points.count < 2 {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray6))
                    .frame(height: 100)
                    .overlay(Text("Move to see signal").font(.caption).foregroundStyle(.tertiary))
            } else {
                Chart(points) { pt in
                    LineMark(x: .value("Time", pt.t), y: .value("Mag", pt.value))
                        .foregroundStyle(Color.green)
                        .lineStyle(StrokeStyle(lineWidth: 1.5))
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { _ in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [3]))
                            .foregroundStyle(Color.secondary.opacity(0.3))
                        AxisValueLabel().font(.system(size: 9)).foregroundStyle(Color.secondary)
                    }
                }
                .frame(height: 100)
                .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }
}

// MARK: - AccelerometerView

struct AccelerometerView: View {
    @Bindable var manager: AccelerometerManager

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if manager.isRecording {
                        LiveChartView(points: manager.liveChartPoints)
                            .padding()
                            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                            .transition(.opacity.combined(with: .move(edge: .top)))
                    }

                    VStack(spacing: 8) {
                        Text("Raw Accelerometer").font(.headline)
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

                    if let status = manager.statusMessage {
                        HStack {
                            Image(systemName: "recordingtape").foregroundStyle(.secondary)
                            Text(status).font(.caption).foregroundStyle(.secondary)
                        }.padding(.horizontal)
                    }

                    Button(action: {
                        withAnimation { manager.isRecording ? manager.stopUpdates() : manager.startUpdates() }
                    }) {
                        Label(
                            manager.isRecording ? "Stop Recording" : "Start Recording",
                            systemImage: manager.isRecording ? "stop.circle.fill" : "record.circle"
                        )
                        .font(.title3.bold()).foregroundStyle(.white).padding()
                        .frame(maxWidth: .infinity)
                        .background(manager.isRecording ? Color.red : Color.accentColor,
                                    in: RoundedRectangle(cornerRadius: 14))
                    }.padding(.horizontal)
                }
                .padding()
            }
            .navigationTitle(manager.userName)
            .animation(.easeInOut(duration: 0.3), value: manager.isRecording)
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

// MARK: - SessionsView

struct SessionsView: View {
    @Bindable var manager: AccelerometerManager
    @State private var shareURL: URL?
    @State private var showShareSheet = false

    var body: some View {
        NavigationStack {
            Group {
                if manager.localSessions.isEmpty {
                    ContentUnavailableView(
                        "No Recordings Yet",
                        systemImage: "waveform",
                        description: Text("Record a session and it will appear here as a CSV file ready to export.")
                    )
                } else {
                    List {
                        ForEach(manager.localSessions) { session in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(session.name)
                                    .font(.subheadline.bold())
                                    .lineLimit(1)
                                HStack {
                                    Text(session.formattedDate)
                                        .font(.caption).foregroundStyle(.secondary)
                                    Spacer()
                                    Text("\(session.sampleCount) samples  •  \(session.formattedSize)")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                            .swipeActions(edge: .trailing) {
                                Button(role: .destructive) {
                                    manager.deleteSession(session)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                            .swipeActions(edge: .leading) {
                                Button {
                                    shareURL = session.fileURL
                                    showShareSheet = true
                                } label: {
                                    Label("Export", systemImage: "square.and.arrow.up")
                                }
                                .tint(.blue)
                            }
                            .contentShape(Rectangle())
                            .onTapGesture {
                                shareURL = session.fileURL
                                showShareSheet = true
                            }
                        }
                    }
                }
            }
            .navigationTitle("Recordings")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { manager.loadLocalSessions() }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .sheet(isPresented: $showShareSheet) {
                if let url = shareURL {
                    ShareSheet(url: url)
                }
            }
        }
    }
}

// MARK: - ShareSheet

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - ContentView

struct ContentView: View {
    @State private var manager = AccelerometerManager()

    var body: some View {
        if manager.userName.isEmpty {
            OnboardingView(manager: manager)
        } else {
            TabView {
                AccelerometerView(manager: manager)
                    .tabItem { Label("Record", systemImage: "waveform.circle") }
                SessionsView(manager: manager)
                    .tabItem { Label("Files", systemImage: "folder") }
                    .onAppear { manager.loadLocalSessions() }
            }
        }
    }
}

@main
struct AccelerometerApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
