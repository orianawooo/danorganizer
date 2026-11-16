using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OsuDan;

public enum HitObjectType
{
	Circle,
	Hold
}

public class HitObject
{
	public double Time { get; set; }
	public int Column { get; set; }
	public HitObjectType Type { get; set; }
	public double EndTime { get; set; }

	public static HitObject? Parse(string line, int keyCount = 4)
	{
		var parts = line.Split(',');
		if (parts.Length < 5) return null;

		try
		{
			var x = double.Parse(parts[0], CultureInfo.InvariantCulture);
			var time = double.Parse(parts[2], CultureInfo.InvariantCulture);
			var typeFlags = int.Parse(parts[3], CultureInfo.InvariantCulture);

			var columnWidth = 512.0 / keyCount;
			var column = Math.Clamp((int)(x / columnWidth), 0, keyCount - 1);
			var isHold = (typeFlags & 128) != 0;
			var endTime = time;

			if (isHold && parts.Length >= 6)
			{
				var lastPart = parts[5];
				var colonIndex = lastPart.IndexOf(':');
				var endStr = colonIndex >= 0 ? lastPart.Substring(0, colonIndex) : lastPart;
				if (double.TryParse(endStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedEnd))
					endTime = parsedEnd;
			}

			return new HitObject
			{
				Time = time,
				Column = column,
				Type = isHold ? HitObjectType.Hold : HitObjectType.Circle,
				EndTime = endTime
			};
		}
		catch
		{
			return null;
		}
	}
}

public class TimingPoint
{
	public double Time { get; set; }
	public double BeatLength { get; set; }
	public bool Uninherited { get; set; } = true;

	public double Bpm => Uninherited && BeatLength > 0 ? 60000.0 / BeatLength : 0;

	public static TimingPoint Parse(string line)
	{
		var parts = line.Split(',');
		if (parts.Length < 2) throw new FormatException();

		return new TimingPoint
		{
			Time = double.Parse(parts[0], CultureInfo.InvariantCulture),
			BeatLength = double.Parse(parts[1], CultureInfo.InvariantCulture),
			Uninherited = parts.Length <= 6 || parts[6] == "1"
		};
	}
}

public class BeatmapData
{
	public string FilePath { get; set; } = string.Empty;
	public int Mode { get; set; }
	public double CircleSize { get; set; } = 4;
	public double OverallDifficulty { get; set; } = 5;
	public string Title { get; set; } = string.Empty;
	public string Artist { get; set; } = string.Empty;
	public string Version { get; set; } = string.Empty;
	public string DisplayName => $"{Artist} - {Title} [{Version}]";

	public List<TimingPoint> TimingPoints { get; set; } = new();
	public List<HitObject> HitObjects { get; set; } = new();

	public static bool QuickCheckIs4K(string filePath)
	{
		try
		{
			using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 4096);
			using var reader = new StreamReader(stream);

			int? mode = null;
			double? circleSize = null;
			string? line;
			var lineCount = 0;

			while ((line = reader.ReadLine()) != null && lineCount++ < 150)
			{
				var trimmed = line.Trim();
				if (trimmed.StartsWith("Mode:", StringComparison.OrdinalIgnoreCase))
				{
					if (int.TryParse(trimmed[5..].Trim(), out var m))
					{
						mode = m;
						if (mode != 3) return false;
					}
				}
				else if (trimmed.StartsWith("CircleSize:", StringComparison.OrdinalIgnoreCase))
				{
					if (double.TryParse(trimmed[11..].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var cs))
					{
						circleSize = cs;
						if (Math.Abs(cs - 4.0) > 0.01) return false;
					}
				}
				else if (trimmed.StartsWith("[HitObjects]", StringComparison.OrdinalIgnoreCase))
				{
					break;
				}

				if (mode == 3 && circleSize.HasValue && Math.Abs(circleSize.Value - 4.0) < 0.01)
					return true;
			}

			return mode == 3 && circleSize.HasValue && Math.Abs(circleSize.Value - 4.0) < 0.01;
		}
		catch
		{
			return false;
		}
	}

	public static BeatmapData Parse(string filePath, byte[] fileBytes)
	{
		var beatmap = new BeatmapData { FilePath = filePath };
		using var stream = new MemoryStream(fileBytes);
		using var reader = new StreamReader(stream, Encoding.UTF8);

		string? line;
		var currentSection = string.Empty;
		var keyCount = 4;

		while ((line = reader.ReadLine()) != null)
		{
			var trimmed = line.Trim();
			if (string.IsNullOrEmpty(trimmed)) continue;

			if (trimmed.StartsWith('[') && trimmed.EndsWith(']'))
			{
				currentSection = trimmed.Trim('[', ']');
				continue;
			}

			switch (currentSection)
			{
				case "General":
					if (trimmed.StartsWith("Mode:", StringComparison.OrdinalIgnoreCase))
					{
						if (int.TryParse(trimmed[5..].Trim(), out var m))
							beatmap.Mode = m;
					}
					break;
				case "Metadata":
					var colonIdx = trimmed.IndexOf(':');
					if (colonIdx > 0)
					{
						var k = trimmed.Substring(0, colonIdx).Trim();
						var v = trimmed.Substring(colonIdx + 1).Trim();
						if (k == "Title") beatmap.Title = v;
						else if (k == "Artist") beatmap.Artist = v;
						else if (k == "Version") beatmap.Version = v;
					}
					break;
				case "Difficulty":
					var diffColon = trimmed.IndexOf(':');
					if (diffColon > 0)
					{
						var k = trimmed.Substring(0, diffColon).Trim();
						var v = trimmed.Substring(diffColon + 1).Trim();
						if (k == "CircleSize" && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var cs))
						{
							beatmap.CircleSize = cs;
							keyCount = (int)cs;
						}
						else if (k == "OverallDifficulty" && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var od))
						{
							beatmap.OverallDifficulty = od;
						}
					}
					break;
				case "TimingPoints":
					if (!trimmed.StartsWith("//"))
					{
						try { beatmap.TimingPoints.Add(TimingPoint.Parse(trimmed)); } catch { }
					}
					break;
				case "HitObjects":
					if (!trimmed.StartsWith("//"))
					{
						var ho = HitObject.Parse(trimmed, keyCount);
						if (ho != null) beatmap.HitObjects.Add(ho);
					}
					break;
			}
		}

		return beatmap;
	}
}

public class MsdScores
{
	[JsonPropertyName("overall")] public float Overall { get; set; }
	[JsonPropertyName("stream")] public float Stream { get; set; }
	[JsonPropertyName("jumpstream")] public float Jumpstream { get; set; }
	[JsonPropertyName("handstream")] public float Handstream { get; set; }
	[JsonPropertyName("stamina")] public float Stamina { get; set; }
	[JsonPropertyName("jackspeed")] public float Jackspeed { get; set; }
	[JsonPropertyName("chordjack")] public float Chordjack { get; set; }
	[JsonPropertyName("technical")] public float Technical { get; set; }

	[JsonIgnore]
	public string DominantSkillset
	{
		get
		{
			var skillsets = new (string Name, float Value)[]
			{
				("Stream", Stream),
				("Jumpstream", Jumpstream),
				("Handstream", Handstream),
				("Stamina", Stamina),
				("Jackspeed", Jackspeed),
				("Chordjack", Chordjack),
				("Technical", Technical)
			};
			return skillsets.MaxBy(s => s.Value).Name;
		}
	}
}

public class MsdResult
{
	[JsonPropertyName("scores")] public MsdScores Scores { get; set; } = new();
}

public class MsdCalculator
{
	private readonly string _executablePath;
	private readonly bool _useWine;

	public MsdCalculator(string? executablePath = null)
	{
		if (string.IsNullOrEmpty(executablePath))
		{
			var baseDir = AppDomain.CurrentDomain.BaseDirectory;
			var currentDir = Directory.GetCurrentDirectory();

			var binaryNames = OperatingSystem.IsWindows()
				? new[] { "msd-calculator-515.exe", "msd-calculator.exe" }
				: OperatingSystem.IsMacOS()
					? new[] { "msd-calculator-515-mac", "msd-calculator-515", "msd-calculator-515.exe" }
					: new[] { "msd-calculator-515", "msd-calculator-515.exe" };

			string? found = null;
			foreach (var name in binaryNames)
			{
				var c1 = Path.Combine(baseDir, "tools", name);
				var c2 = Path.Combine(currentDir, "tools", name);
				var c3 = Path.Combine(baseDir, name);
				var c4 = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "tools", name));

				if (File.Exists(c1)) { found = c1; break; }
				if (File.Exists(c2)) { found = c2; break; }
				if (File.Exists(c3)) { found = c3; break; }
				if (File.Exists(c4)) { found = c4; break; }
			}

			if (found != null) _executablePath = found;
			else throw new FileNotFoundException("Msd calculator binary not found in tools directory.");
		}
		else
		{
			if (!File.Exists(executablePath))
				throw new FileNotFoundException($"msd-calculator not found at: {executablePath}");
			_executablePath = executablePath;
		}

		if (!OperatingSystem.IsWindows())
		{
			if (_executablePath.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
			{
				_useWine = true;
			}
			else
			{
				try
				{
					File.SetUnixFileMode(_executablePath,
						UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
						UnixFileMode.GroupRead | UnixFileMode.GroupExecute |
						UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
				}
				catch { }
			}
		}
	}

	public async Task<MsdScores?> CalculateAsync(string beatmapPath, int timeoutMs = 15000)
	{
		if (!File.Exists(beatmapPath)) return null;

		var startInfo = new ProcessStartInfo
		{
			FileName = _useWine ? "wine" : _executablePath,
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			UseShellExecute = false,
			CreateNoWindow = true
		};

		if (_useWine) startInfo.ArgumentList.Add(_executablePath);
		startInfo.ArgumentList.Add(beatmapPath);
		startInfo.ArgumentList.Add("--rate");
		startInfo.ArgumentList.Add("1.0");

		using var process = new Process { StartInfo = startInfo };
		try
		{
			process.Start();
			var stdoutTask = process.StandardOutput.ReadToEndAsync();
			var stderrTask = process.StandardError.ReadToEndAsync();

			var completed = await Task.Run(() => process.WaitForExit(timeoutMs));
			if (!completed)
			{
				try { process.Kill(); } catch { }
				return null;
			}

			if (process.ExitCode != 0) return null;

			var output = await stdoutTask;
			if (string.IsNullOrWhiteSpace(output)) return null;

			var jsonStart = output.IndexOf('{');
			var jsonEnd = output.LastIndexOf('}');
			if (jsonStart < 0 || jsonEnd < jsonStart) return null;

			var jsonPayload = output.Substring(jsonStart, jsonEnd - jsonStart + 1);
			var parsed = JsonSerializer.Deserialize<MsdResult>(jsonPayload);
			return parsed?.Scores;
		}
		catch
		{
			return null;
		}
	}
}

public class InterludeCalculator
{
	private const float CurvePower = 0.6f;
	private const float CurveScale = 0.4056f;
	private const float MostImportantNotes = 2500.0f;
	private const float JackCurveCutoff = 230.0f;
	private const float StreamCurveCutoff = 10.0f;
	private const float StreamCurveCutoff2 = 10.0f;
	private const float OhtNerf = 3.0f;
	private const float StreamScale = 6.0f;
	private const float StreamPow = 0.5f;
	private const float StrainScale = 0.01626f;
	private const float StrainTimeCap = 200.0f;

	public static double Calculate(BeatmapData beatmap, float rate = 1.0f)
	{
		if (beatmap.HitObjects.Count == 0 || beatmap.TimingPoints.Count == 0)
			return 0.0;

		var noteRows = beatmap.HitObjects
			.GroupBy(h => h.Time)
			.OrderBy(g => g.Key)
			.Select(g => new
			{
				Time = (float)g.Key,
				Notes = g.Select(h => new { h.Column, h.Type }).ToList()
			})
			.ToList();

		if (noteRows.Count == 0) return 0.0;

		var uninherited = beatmap.TimingPoints.Where(t => t.Uninherited).OrderBy(t => t.Time).ToList();
		var lastNoteInColumn = new float[4];
		var strainValues = new float[4];
		var strainDataPoints = new List<float>();

		foreach (var row in noteRows)
		{
			var time = row.Time / rate;

			TimingPoint? activePoint = null;
			foreach (var tp in uninherited)
			{
				if (tp.Time <= row.Time) activePoint = tp;
				else break;
			}
			var bpm = (float)((activePoint ?? (uninherited.Count > 0 ? uninherited[0] : null))?.Bpm ?? 120.0);

			var noteDifficulties = new float[4];
			var rowStrains = new float[4];

			for (var k = 0; k < 4; k++)
			{
				var hasNote = row.Notes.Any(n => n.Column == k && (n.Type == HitObjectType.Circle || n.Type == HitObjectType.Hold));
				if (!hasNote) continue;

				var jackDelta = time - lastNoteInColumn[k];
				var j = jackDelta > 0 ? Math.Min(15000.0f / jackDelta, JackCurveCutoff) : 0.0f;

				var handLo = k < 2 ? 0 : 2;
				var handHi = k < 2 ? 1 : 3;
				var sl = 0.0f;
				var sr = 0.0f;

				for (var handK = handLo; handK <= handHi; handK++)
				{
					if (handK == k) continue;
					var trillDelta = time - lastNoteInColumn[handK];
					if (trillDelta > 0)
					{
						var trillV = (300.0f / (0.02f * trillDelta) - 300.0f / (float)Math.Pow(0.02f * trillDelta, StreamCurveCutoff) / StreamCurveCutoff2);
						trillV = Math.Max(0.0f, trillV);

						var ratio = jackDelta / trillDelta;
						var logRatio = (float)Math.Log(ratio, 2.0);
						var comp = Math.Min(1.0f, (float)Math.Sqrt(Math.Max(0.0f, logRatio)));
						trillV *= comp;

						if (handK < k) sl = Math.Max(sl, trillV);
						else sr = Math.Max(sr, trillV);
					}
				}

				noteDifficulties[k] = (float)Math.Pow(
					Math.Pow(StreamScale * (float)Math.Pow(sl, StreamPow), OhtNerf) +
					Math.Pow(StreamScale * (float)Math.Pow(sr, StreamPow), OhtNerf) +
					Math.Pow(j, OhtNerf),
					1.0f / OhtNerf
				);

				var input = noteDifficulties[k];
				var delta = Math.Max(0.0f, jackDelta);

				var decayRate = (float)Math.Log(0.5) / 1575.0f;
				var decay = (float)Math.Exp(decayRate * Math.Min(StrainTimeCap, delta));
				var timeCapDecay = delta > StrainTimeCap ? (float)Math.Exp(decayRate * (delta - StrainTimeCap)) : 1.0f;
				var a = strainValues[k] * timeCapDecay;
				var b = input * input * StrainScale;
				strainValues[k] = b - (b - a) * decay;
				rowStrains[k] = strainValues[k];

				lastNoteInColumn[k] = time;
			}

			foreach (var strain in rowStrains)
			{
				if (strain > 0.0f) strainDataPoints.Add(strain);
			}
		}

		var dataArray = strainDataPoints.Where(x => x > 0.0f).OrderBy(x => x).ToArray();
		var length = (float)dataArray.Length;
		if (dataArray.Length == 0) return 0.0;

		var weight = 0.0f;
		var total = 0.0f;
		for (var i = 0; i < dataArray.Length; i++)
		{
			var position = (i + MostImportantNotes - length) / MostImportantNotes;
			var x = Math.Max(0.0f, position);
			var w = 0.002f + (float)Math.Pow(x, 4.0);
			weight += w;
			total += dataArray[i] * w;
		}

		if (weight <= 0.0f) return 0.0;

		var result = (float)Math.Pow(total / weight, CurvePower) * CurveScale;
		return float.IsFinite(result) ? result : 0.0;
	}
}

public class SunnyCalculator
{
	public static double Calculate(BeatmapData beatmap, float rate = 1.0f)
	{
		if (beatmap.HitObjects.Count == 0) return -1.0;

		try
		{
			var noteSeq = new List<(int k, int h, int t)>();
			foreach (var obj in beatmap.HitObjects)
			{
				var k = obj.Column;
				var h = (int)obj.Time;
				var t = obj.Type == HitObjectType.Hold ? (int)obj.EndTime : -1;

				if (Math.Abs(rate - 1.5f) < 0.01f)
				{
					h = (int)Math.Floor(h * 2.0 / 3.0);
					if (t >= 0) t = (int)Math.Floor(t * 2.0 / 3.0);
				}
				else if (Math.Abs(rate - 0.75f) < 0.01f)
				{
					h = (int)Math.Floor(h * 4.0 / 3.0);
					if (t >= 0) t = (int)Math.Floor(t * 4.0 / 3.0);
				}

				noteSeq.Add((k, h, t));
			}

			var x = 0.3 * Math.Pow((64.5 - Math.Ceiling(beatmap.OverallDifficulty * 3)) / 500.0, 0.5);
			x = Math.Min(x, 0.6 * (x - 0.09) + 0.09);

			noteSeq = noteSeq.OrderBy(n => n.h).ThenBy(n => n.k).ToList();
			if (noteSeq.Count == 0) return -1.0;

			var K = (int)beatmap.CircleSize;
			var T = Math.Max(noteSeq.Max(n => n.h), noteSeq.Max(n => n.t)) + 1;

			var noteDict = new Dictionary<int, List<(int k, int h, int t)>>();
			foreach (var n in noteSeq)
			{
				if (!noteDict.ContainsKey(n.k)) noteDict[n.k] = new List<(int k, int h, int t)>();
				noteDict[n.k].Add(n);
			}

			var noteSeqByColumn = noteDict.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
			var lnSeq = noteSeq.Where(n => n.t >= 0).ToList();
			var tailSeq = lnSeq.OrderBy(n => n.t).ToList();

			var (allCorners, baseCorners, aCorners) = GetCorners(T, noteSeq);
			var keyUsage = GetKeyUsage(K, T, noteSeq, baseCorners);

			var activeColumns = new List<List<int>>();
			for (var i = 0; i < baseCorners.Length; i++)
			{
				var cols = new List<int>();
				for (var col = 0; col < K; col++)
					if (keyUsage[col][i]) cols.Add(col);
				activeColumns.Add(cols);
			}

			var keyUsage400 = GetKeyUsage400(K, T, noteSeq, baseCorners);
			var anchor = ComputeAnchor(K, keyUsage400, baseCorners);

			var (deltaKs, jbar) = ComputeJbar(K, x, noteSeqByColumn, baseCorners);
			jbar = InterpValues(allCorners, baseCorners, jbar);

			var xbar = ComputeXbar(K, x, noteSeqByColumn, activeColumns, baseCorners);
			xbar = InterpValues(allCorners, baseCorners, xbar);

			var lnRep = LnBodiesCountSparse(lnSeq, T);
			var pbar = ComputePbar(x, noteSeq, lnRep, anchor, baseCorners);
			pbar = InterpValues(allCorners, baseCorners, pbar);

			var abar = ComputeAbar(K, activeColumns, deltaKs, aCorners, baseCorners);
			abar = InterpValues(allCorners, aCorners, abar);

			var rbar = ComputeRbar(x, noteSeqByColumn, tailSeq, baseCorners);
			rbar = InterpValues(allCorners, baseCorners, rbar);

			var (cStep, ksStep) = ComputeCAndKs(K, noteSeq, keyUsage, baseCorners);
			var cArr = StepInterp(allCorners, baseCorners, cStep);
			var ksArr = StepInterp(allCorners, baseCorners, ksStep);

			var sAll = new double[allCorners.Length];
			var tAll = new double[allCorners.Length];
			var dAll = new double[allCorners.Length];

			for (var i = 0; i < allCorners.Length; i++)
			{
				var term1 = Math.Pow(abar[i], 3.0 / ksArr[i]) * Math.Min(jbar[i], 8 + 0.85 * jbar[i]);
				var term2 = Math.Pow(abar[i], 2.0 / 3.0) * (0.8 * pbar[i] + rbar[i] * 35.0 / (cArr[i] + 8));

				sAll[i] = Math.Pow(0.4 * Math.Pow(term1, 1.5) + 0.6 * Math.Pow(term2, 1.5), 2.0 / 3.0);
				tAll[i] = Math.Pow(abar[i], 3.0 / ksArr[i]) * xbar[i] / (xbar[i] + sAll[i] + 1);
				dAll[i] = 2.7 * Math.Pow(sAll[i], 0.5) * Math.Pow(tAll[i], 1.5) + sAll[i] * 0.27;
			}

			var gaps = new double[allCorners.Length];
			gaps[0] = (allCorners[1] - allCorners[0]) / 2.0;
			gaps[^1] = (allCorners[^1] - allCorners[^2]) / 2.0;
			for (var i = 1; i < allCorners.Length - 1; i++) gaps[i] = (allCorners[i + 1] - allCorners[i - 1]) / 2.0;

			var effectiveWeights = new double[allCorners.Length];
			for (var i = 0; i < allCorners.Length; i++) effectiveWeights[i] = cArr[i] * gaps[i];

			var sortedIndices = Enumerable.Range(0, dAll.Length).OrderBy(i => dAll[i]).ToArray();
			var dSorted = sortedIndices.Select(i => dAll[i]).ToArray();
			var wSorted = sortedIndices.Select(i => effectiveWeights[i]).ToArray();

			var cumWeights = new double[wSorted.Length];
			cumWeights[0] = wSorted[0];
			for (var i = 1; i < wSorted.Length; i++) cumWeights[i] = cumWeights[i - 1] + wSorted[i];

			var totalWeight = cumWeights[^1];
			var normCumWeights = cumWeights.Select(w => w / totalWeight).ToArray();

			var targetPercentiles = new[] { 0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815 };
			var indices = new int[targetPercentiles.Length];

			for (var i = 0; i < targetPercentiles.Length; i++)
			{
				indices[i] = SearchSortedLeft(normCumWeights, targetPercentiles[i]);
				if (indices[i] >= dSorted.Length) indices[i] = dSorted.Length - 1;
			}

			var p93 = (dSorted[indices[0]] + dSorted[indices[1]] + dSorted[indices[2]] + dSorted[indices[3]]) / 4.0;
			var p83 = (dSorted[indices[4]] + dSorted[indices[5]] + dSorted[indices[6]] + dSorted[indices[7]]) / 4.0;

			double sumD5W = 0, sumW = 0;
			for (var i = 0; i < dSorted.Length; i++)
			{
				sumD5W += Math.Pow(dSorted[i], 5) * wSorted[i];
				sumW += wSorted[i];
			}

			var weightedMean = Math.Pow(sumD5W / sumW, 0.2);
			var sr = 0.88 * p93 * 0.25 + 0.94 * p83 * 0.2 + weightedMean * 0.55;

			double totalNotes = noteSeq.Count;
			foreach (var (_, h, t) in lnSeq) totalNotes += 0.5 * Math.Min(t - h, 1000) / 200.0;

			sr *= totalNotes / (totalNotes + 60);
			sr = sr <= 9 ? sr : 9 + (sr - 9) * (1.0 / 1.2);
			sr *= 0.975;

			return sr;
		}
		catch
		{
			return -1.0;
		}
	}

	private static int SearchSortedLeft(double[] arr, double val)
	{
		int lo = 0, hi = arr.Length;
		while (lo < hi)
		{
			var mid = (lo + hi) / 2;
			if (arr[mid] < val) lo = mid + 1;
			else hi = mid;
		}
		return lo;
	}

	private static int SearchSortedRight(double[] arr, double val)
	{
		int lo = 0, hi = arr.Length;
		while (lo < hi)
		{
			var mid = (lo + hi) / 2;
			if (arr[mid] <= val) lo = mid + 1;
			else hi = mid;
		}
		return lo;
	}

	private static double[] CumulativeSum(double[] x, double[] f)
	{
		var F = new double[x.Length];
		for (var i = 1; i < x.Length; i++) F[i] = F[i - 1] + f[i - 1] * (x[i] - x[i - 1]);
		return F;
	}

	private static double QueryCumsum(double q, double[] x, double[] F, double[] f)
	{
		if (q <= x[0]) return 0.0;
		if (q >= x[^1]) return F[^1];
		var i = SearchSortedLeft(x, q) - 1;
		if (i < 0) i = 0;
		return F[i] + f[i] * (q - x[i]);
	}

	private static double[] SmoothOnCorners(double[] x, double[] f, double window, double scale = 1.0, string mode = "sum")
	{
		var F = CumulativeSum(x, f);
		var g = new double[f.Length];
		for (var i = 0; i < x.Length; i++)
		{
			var s = x[i];
			var a = Math.Max(s - window, x[0]);
			var b = Math.Min(s + window, x[^1]);
			var val = QueryCumsum(b, x, F, f) - QueryCumsum(a, x, F, f);
			g[i] = mode == "avg" ? (b - a > 0 ? val / (b - a) : 0.0) : scale * val;
		}
		return g;
	}

	private static double[] InterpValues(double[] newX, double[] oldX, double[] oldVals)
	{
		var newVals = new double[newX.Length];
		for (var i = 0; i < newX.Length; i++)
		{
			var xVal = newX[i];
			if (xVal <= oldX[0]) newVals[i] = oldVals[0];
			else if (xVal >= oldX[^1]) newVals[i] = oldVals[^1];
			else
			{
				var pos = SearchSortedLeft(oldX, xVal);
				if (pos == 0) newVals[i] = oldVals[0];
				else if (pos >= oldX.Length) newVals[i] = oldVals[^1];
				else
				{
					var x0 = oldX[pos - 1];
					var x1 = oldX[pos];
					var y0 = oldVals[pos - 1];
					var y1 = oldVals[pos];
					var ratio = (xVal - x0) / (x1 - x0);
					newVals[i] = y0 + ratio * (y1 - y0);
				}
			}
		}
		return newVals;
	}

	private static double[] StepInterp(double[] newX, double[] oldX, double[] oldVals)
	{
		var result = new double[newX.Length];
		for (var i = 0; i < newX.Length; i++)
		{
			var idx = SearchSortedRight(oldX, newX[i]) - 1;
			idx = Math.Clamp(idx, 0, oldVals.Length - 1);
			result[i] = oldVals[idx];
		}
		return result;
	}

	private static (double[] allCorners, double[] baseCorners, double[] aCorners) GetCorners(int T, List<(int k, int h, int t)> noteSeq)
	{
		var cornersBase = new HashSet<double>();
		foreach (var (_, h, t) in noteSeq)
		{
			cornersBase.Add(h);
			if (t >= 0) cornersBase.Add(t);
		}

		var toAdd = new List<double>();
		foreach (var s in cornersBase)
		{
			toAdd.Add(s + 501);
			toAdd.Add(s - 499);
			toAdd.Add(s + 1);
		}

		foreach (var s in toAdd) cornersBase.Add(s);
		cornersBase.Add(0);
		cornersBase.Add(T);
		var baseCorners = cornersBase.Where(s => s >= 0 && s <= T).OrderBy(s => s).ToArray();

		var cornersA = new HashSet<double>();
		foreach (var (_, h, t) in noteSeq)
		{
			cornersA.Add(h);
			if (t >= 0) cornersA.Add(t);
		}

		toAdd.Clear();
		foreach (var s in cornersA)
		{
			toAdd.Add(s + 1000);
			toAdd.Add(s - 1000);
		}
		foreach (var s in toAdd) cornersA.Add(s);
		cornersA.Add(0);
		cornersA.Add(T);

		var aCorners = cornersA.Where(s => s >= 0 && s <= T).OrderBy(s => s).ToArray();
		var allCorners = baseCorners.Union(aCorners).OrderBy(s => s).ToArray();
		return (allCorners, baseCorners, aCorners);
	}

	private static bool[][] GetKeyUsage(int K, int T, List<(int k, int h, int t)> noteSeq, double[] baseCorners)
	{
		var keyUsage = new bool[K][];
		for (var col = 0; col < K; col++) keyUsage[col] = new bool[baseCorners.Length];

		foreach (var (col, h, t) in noteSeq)
		{
			double startTime = Math.Max(h - 150, 0);
			double endTime = t < 0 ? h + 150 : Math.Min(t + 150, T - 1);
			var leftIdx = SearchSortedLeft(baseCorners, startTime);
			var rightIdx = SearchSortedLeft(baseCorners, endTime);

			for (var idx = leftIdx; idx < rightIdx; idx++) keyUsage[col][idx] = true;
		}
		return keyUsage;
	}

	private static double[][] GetKeyUsage400(int K, int T, List<(int k, int h, int t)> noteSeq, double[] baseCorners)
	{
		var keyUsage400 = new double[K][];
		for (var col = 0; col < K; col++) keyUsage400[col] = new double[baseCorners.Length];

		foreach (var (col, h, t) in noteSeq)
		{
			double startTime = Math.Max(h, 0);
			double endTime = t < 0 ? h : Math.Min(t, T - 1);
			var left400Idx = SearchSortedLeft(baseCorners, startTime - 400);
			var leftIdx = SearchSortedLeft(baseCorners, startTime);
			var rightIdx = SearchSortedLeft(baseCorners, endTime);
			var right400Idx = SearchSortedLeft(baseCorners, endTime + 400);

			for (var idx = leftIdx; idx < rightIdx; idx++)
				keyUsage400[col][idx] += 3.75 + Math.Min(endTime - startTime, 1500) / 150.0;

			for (var idx = left400Idx; idx < leftIdx; idx++)
			{
				var dist = baseCorners[idx] - startTime;
				keyUsage400[col][idx] += 3.75 - 3.75 / 160000.0 * (dist * dist);
			}

			for (var idx = rightIdx; idx < right400Idx; idx++)
			{
				var dist = Math.Abs(baseCorners[idx] - endTime);
				keyUsage400[col][idx] += 3.75 - 3.75 / 160000.0 * (dist * dist);
			}
		}
		return keyUsage400;
	}

	private static double[] ComputeAnchor(int K, double[][] keyUsage400, double[] baseCorners)
	{
		var anchor = new double[baseCorners.Length];
		for (var idx = 0; idx < baseCorners.Length; idx++)
		{
			var counts = new double[K];
			for (var col = 0; col < K; col++) counts[col] = keyUsage400[col][idx];
			Array.Sort(counts);
			Array.Reverse(counts);

			var nonzeroCounts = counts.Where(c => c != 0).ToArray();
			if (nonzeroCounts.Length > 1)
			{
				double walk = 0, maxWalk = 0;
				for (var i = 0; i < nonzeroCounts.Length - 1; i++)
				{
					var ratio = nonzeroCounts[i + 1] / nonzeroCounts[i];
					walk += nonzeroCounts[i] * (1 - 4 * Math.Pow(0.5 - ratio, 2));
					maxWalk += nonzeroCounts[i];
				}
				anchor[idx] = walk / maxWalk;
			}
		}

		for (var i = 0; i < anchor.Length; i++)
			anchor[i] = 1 + Math.Min(anchor[i] - 0.18, 5 * Math.Pow(anchor[i] - 0.22, 3));

		return anchor;
	}

	private static (double[][] deltaKs, double[] jbar) ComputeJbar(int K, double x, List<List<(int k, int h, int t)>> noteSeqByColumn, double[] baseCorners)
	{
		var jKs = new double[K][];
		var deltaKs = new double[K][];
		for (var col = 0; col < K; col++)
		{
			jKs[col] = new double[baseCorners.Length];
			deltaKs[col] = new double[baseCorners.Length];
			Array.Fill(deltaKs[col], 1e9);
		}

		for (var col = 0; col < K; col++)
		{
			if (col >= noteSeqByColumn.Count) continue;
			var notes = noteSeqByColumn[col];

			for (var i = 0; i < notes.Count - 1; i++)
			{
				double start = notes[i].h;
				double end = notes[i + 1].h;
				var leftIdx = SearchSortedLeft(baseCorners, start);
				var rightIdx = SearchSortedLeft(baseCorners, end);
				if (leftIdx >= rightIdx) continue;

				var delta = 0.001 * (end - start);
				var val = Math.Pow(delta, -1) * Math.Pow(delta + 0.11 * Math.Pow(x, 0.25), -1);
				var nerfer = 1 - 7e-5 * Math.Pow(0.15 + Math.Abs(delta - 0.08), -4);
				var jVal = val * nerfer;

				for (var idx = leftIdx; idx < rightIdx; idx++)
				{
					jKs[col][idx] = jVal;
					deltaKs[col][idx] = delta;
				}
			}
		}

		var jbarKs = new double[K][];
		for (var col = 0; col < K; col++) jbarKs[col] = SmoothOnCorners(baseCorners, jKs[col], 500, 0.001, "sum");

		var jbar = new double[baseCorners.Length];
		for (var i = 0; i < baseCorners.Length; i++)
		{
			double num = 0, den = 0;
			for (var col = 0; col < K; col++)
			{
				var v = jbarKs[col][i];
				var w = 1.0 / deltaKs[col][i];
				num += Math.Pow(Math.Max(v, 0), 5) * w;
				den += w;
			}
			jbar[i] = Math.Pow(num / Math.Max(1e-9, den), 0.2);
		}

		return (deltaKs, jbar);
	}

	private static double[] ComputeXbar(int K, double x, List<List<(int k, int h, int t)>> noteSeqByColumn, List<List<int>> activeColumns, double[] baseCorners)
	{
		var crossMatrix = new double[][]
		{
			new[] { -1.0 },
			new[] { 0.075, 0.075 },
			new[] { 0.125, 0.05, 0.125 },
			new[] { 0.125, 0.125, 0.125, 0.125 },
			new[] { 0.175, 0.25, 0.05, 0.25, 0.175 },
			new[] { 0.175, 0.25, 0.175, 0.175, 0.25, 0.175 },
			new[] { 0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225 },
			new[] { 0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225 },
			new[] { 0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275 },
			new[] { 0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275 },
			new[] { 0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325 }
		};

		var xKs = new double[K + 1][];
		var fastCross = new double[K + 1][];
		for (var col = 0; col <= K; col++)
		{
			xKs[col] = new double[baseCorners.Length];
			fastCross[col] = new double[baseCorners.Length];
		}

		var crossCoeff = crossMatrix[K];
		for (var col = 0; col <= K; col++)
		{
			List<(int k, int h, int t)> notesInPair;
			if (col == 0) notesInPair = noteSeqByColumn.Count > 0 ? noteSeqByColumn[0] : new List<(int k, int h, int t)>();
			else if (col == K) notesInPair = noteSeqByColumn.Count > 0 ? noteSeqByColumn[K - 1] : new List<(int k, int h, int t)>();
			else
			{
				var l1 = col - 1 < noteSeqByColumn.Count ? noteSeqByColumn[col - 1] : new List<(int k, int h, int t)>();
				var l2 = col < noteSeqByColumn.Count ? noteSeqByColumn[col] : new List<(int k, int h, int t)>();
				notesInPair = l1.Concat(l2).OrderBy(n => n.h).ToList();
			}

			for (var i = 1; i < notesInPair.Count; i++)
			{
				double start = notesInPair[i - 1].h;
				double end = notesInPair[i].h;
				var idxStart = SearchSortedLeft(baseCorners, start);
				var idxEnd = SearchSortedLeft(baseCorners, end);
				if (idxStart >= idxEnd) continue;

				var delta = 0.001 * (notesInPair[i].h - notesInPair[i - 1].h);
				var val = 0.16 * Math.Pow(Math.Max(x, delta), -2);

				var prevNotInStart = !activeColumns[idxStart].Contains(col - 1);
				var prevNotInEnd = !activeColumns[idxEnd < activeColumns.Count ? idxEnd : activeColumns.Count - 1].Contains(col - 1);
				var currNotInStart = !activeColumns[idxStart].Contains(col);
				var currNotInEnd = !activeColumns[idxEnd < activeColumns.Count ? idxEnd : activeColumns.Count - 1].Contains(col);

				if ((prevNotInStart && prevNotInEnd) || (currNotInStart && currNotInEnd))
					val *= 1 - crossCoeff[col];

				for (var idx = idxStart; idx < idxEnd; idx++)
				{
					xKs[col][idx] = val;
					fastCross[col][idx] = Math.Max(0, 0.4 * Math.Pow(Math.Max(Math.Max(delta, 0.06), 0.75 * x), -2) - 80);
				}
			}
		}

		var xBase = new double[baseCorners.Length];
		for (var i = 0; i < baseCorners.Length; i++)
		{
			double sum1 = 0;
			for (var col = 0; col <= K; col++) sum1 += xKs[col][i] * crossCoeff[col];
			double sum2 = 0;
			for (var col = 0; col < K; col++) sum2 += Math.Sqrt(fastCross[col][i] * crossCoeff[col] * fastCross[col + 1][i] * crossCoeff[col + 1]);
			xBase[i] = sum1 + sum2;
		}

		return SmoothOnCorners(baseCorners, xBase, 500, 0.001, "sum");
	}

	private static (List<double> points, List<double> cumsum, List<double> values) LnBodiesCountSparse(List<(int k, int h, int t)> lnSeq, int T)
	{
		var diff = new Dictionary<double, double>();
		foreach (var (_, h, t) in lnSeq)
		{
			double t0 = Math.Min(h + 60, t);
			double t1 = Math.Min(h + 120, t);
			diff.TryAdd(t0, 0); diff[t0] += 1.3;
			diff.TryAdd(t1, 0); diff[t1] += -0.3;
			diff.TryAdd(t, 0); diff[t] -= 1;
		}

		var pointsSet = new HashSet<double> { 0, T };
		pointsSet.UnionWith(diff.Keys);
		var points = pointsSet.OrderBy(p => p).ToList();
		var values = new List<double>();
		var cumsum = new List<double> { 0 };
		var curr = 0.0;

		for (var i = 0; i < points.Count - 1; i++)
		{
			var time = points[i];
			if (diff.TryGetValue(time, out var change)) curr += change;
			var v = Math.Min(curr, 2.5 + 0.5 * curr);
			values.Add(v);
			cumsum.Add(cumsum[^1] + (points[i + 1] - points[i]) * v);
		}

		return (points, cumsum, values);
	}

	private static double LnSum(double a, double b, (List<double> points, List<double> cumsum, List<double> values) lnRep)
	{
		var (points, cumsum, values) = lnRep;
		var i = Math.Max(0, SearchSortedRight(points.ToArray(), a) - 1);
		var j = Math.Max(0, SearchSortedRight(points.ToArray(), b) - 1);
		if (values.Count == 0) return 0;
		i = Math.Min(i, values.Count - 1);
		j = Math.Min(j, values.Count - 1);
		if (i == j) return (b - a) * values[i];

		var total = (points[i + 1] - a) * values[i];
		total += cumsum[j] - cumsum[i + 1];
		total += (b - points[j]) * values[j];
		return total;
	}

	private static double[] ComputePbar(double x, List<(int k, int h, int t)> noteSeq, (List<double>, List<double>, List<double>) lnRep, double[] anchor, double[] baseCorners)
	{
		var pStep = new double[baseCorners.Length];
		for (var i = 0; i < noteSeq.Count - 1; i++)
		{
			double hL = noteSeq[i].h;
			double hR = noteSeq[i + 1].h;
			var deltaTime = hR - hL;

			if (deltaTime < 1e-9)
			{
				var spike = 1000 * Math.Pow(0.02 * (4 / x - 24), 0.25);
				var leftIdx = SearchSortedLeft(baseCorners, hL);
				var rightIdx = SearchSortedRight(baseCorners, hL);
				for (var idx = leftIdx; idx < rightIdx; idx++)
					if (idx >= 0 && idx < pStep.Length) pStep[idx] += spike;
				continue;
			}

			var start = SearchSortedLeft(baseCorners, hL);
			var end = SearchSortedLeft(baseCorners, hR);
			if (start >= end) continue;

			var delta = 0.001 * deltaTime;
			var v = 1 + 6 * 0.001 * LnSum(hL, hR, lnRep);
			var nps = 7.5 / delta;
			var bVal = 160 < nps && nps < 360 ? 1 + 1.7e-7 * (nps - 160) * Math.Pow(nps - 360, 2) : 1.0;

			double inc = delta < 2 * x / 3
				? Math.Pow(delta, -1) * Math.Pow(0.08 * Math.Pow(x, -1) * (1 - 24 * Math.Pow(x, -1) * Math.Pow(delta - x / 2, 2)), 0.25) * Math.Max(bVal, v)
				: Math.Pow(delta, -1) * Math.Pow(0.08 * Math.Pow(x, -1) * (1 - 24 * Math.Pow(x, -1) * Math.Pow(x / 6, 2)), 0.25) * Math.Max(bVal, v);

			for (var idx = start; idx < end; idx++)
				pStep[idx] += Math.Min(inc * anchor[idx], Math.Max(inc, inc * 2 - 10));
		}

		return SmoothOnCorners(baseCorners, pStep, 500, 0.001, "sum");
	}

	private static double[] ComputeAbar(int K, List<List<int>> activeColumns, double[][] deltaKs, double[] aCorners, double[] baseCorners)
	{
		var dks = new double[K - 1][];
		for (var col = 0; col < K - 1; col++) dks[col] = new double[baseCorners.Length];

		for (var i = 0; i < baseCorners.Length; i++)
		{
			var cols = activeColumns[i];
			for (var j = 0; j < cols.Count - 1; j++)
			{
				var k0 = cols[j];
				var k1 = cols[j + 1];
				if (k0 < K - 1 && k0 < deltaKs.Length && k1 < deltaKs.Length)
					dks[k0][i] = Math.Abs(deltaKs[k0][i] - deltaKs[k1][i]) + 0.4 * Math.Max(0, Math.Max(deltaKs[k0][i], deltaKs[k1][i]) - 0.11);
			}
		}

		var aStep = new double[aCorners.Length];
		Array.Fill(aStep, 1.0);

		for (var i = 0; i < aCorners.Length; i++)
		{
			var s = aCorners[i];
			var idx = Math.Min(SearchSortedLeft(baseCorners, s), baseCorners.Length - 1);
			var cols = activeColumns[idx];
			for (var j = 0; j < cols.Count - 1; j++)
			{
				var k0 = cols[j];
				var k1 = cols[j + 1];
				if (k0 >= K - 1 || k0 >= dks.Length) continue;
				var dVal = dks[k0][idx];
				if (dVal < 0.02) aStep[i] *= Math.Min(0.75 + 0.5 * Math.Max(deltaKs[k0][idx], deltaKs[k1][idx]), 1);
				else if (dVal < 0.07) aStep[i] *= Math.Min(0.65 + 5 * dVal + 0.5 * Math.Max(deltaKs[k0][idx], deltaKs[k1][idx]), 1);
			}
		}

		return SmoothOnCorners(aCorners, aStep, 250, 1.0, "avg");
	}

	private static double[] ComputeRbar(double x, List<List<(int k, int h, int t)>> noteSeqByColumn, List<(int k, int h, int t)> tailSeq, double[] baseCorners)
	{
		var rStep = new double[baseCorners.Length];
		if (tailSeq.Count == 0) return SmoothOnCorners(baseCorners, rStep, 500, 0.001, "sum");

		var timesByColumn = new Dictionary<int, List<int>>();
		for (var i = 0; i < noteSeqByColumn.Count; i++)
			timesByColumn[i] = noteSeqByColumn[i].Select(n => n.h).ToList();

		var iList = new double[tailSeq.Count];
		for (var i = 0; i < tailSeq.Count; i++)
		{
			var (k, hI, tI) = tailSeq[i];
			if (!timesByColumn.ContainsKey(k) || k >= noteSeqByColumn.Count)
			{
				iList[i] = 0;
				continue;
			}

			var times = timesByColumn[k];
			var idx = SearchSortedLeft(times.Select(t => (double)t).ToArray(), hI);
			var nextNoteH = idx + 1 < noteSeqByColumn[k].Count ? noteSeqByColumn[k][idx + 1].h : 1000000000;
			var iH = 0.001 * Math.Abs(tI - hI - 80) / x;
			var iT = 0.001 * Math.Abs(nextNoteH - tI - 80) / x;
			iList[i] = 2.0 / (2 + Math.Exp(-5 * (iH - 0.75)) + Math.Exp(-5 * (iT - 0.75)));
		}

		for (var i = 0; i < tailSeq.Count - 1; i++)
		{
			var leftIdx = SearchSortedLeft(baseCorners, tailSeq[i].t);
			var rightIdx = SearchSortedLeft(baseCorners, tailSeq[i + 1].t);
			if (leftIdx >= rightIdx) continue;

			var deltaR = 0.001 * (tailSeq[i + 1].t - tailSeq[i].t);
			var rValue = 0.08 * Math.Pow(deltaR, -0.5) * Math.Pow(x, -1) * (1 + 0.8 * (iList[i] + iList[i + 1]));
			for (var idx = leftIdx; idx < rightIdx; idx++) rStep[idx] = rValue;
		}

		return SmoothOnCorners(baseCorners, rStep, 500, 0.001, "sum");
	}

	private static (double[] cStep, double[] ksStep) ComputeCAndKs(int K, List<(int k, int h, int t)> noteSeq, bool[][] keyUsage, double[] baseCorners)
	{
		var noteHitTimes = noteSeq.Select(n => (double)n.h).OrderBy(t => t).ToArray();
		var cStep = new double[baseCorners.Length];
		for (var i = 0; i < baseCorners.Length; i++)
		{
			var highIdx = SearchSortedLeft(noteHitTimes, baseCorners[i] + 500);
			var lowIdx = SearchSortedLeft(noteHitTimes, baseCorners[i] - 500);
			cStep[i] = highIdx - lowIdx;
		}

		var ksStep = new double[baseCorners.Length];
		for (var i = 0; i < baseCorners.Length; i++)
		{
			var keyCount = 0;
			for (var col = 0; col < K; col++)
				if (keyUsage[col][i]) keyCount++;
			ksStep[i] = Math.Max(keyCount, 1);
		}

		return (cStep, ksStep);
	}
}

public class DanClassification
{
	public int DanIndex { get; set; }
	public string DanName { get; set; } = string.Empty;
	public string DominantSkillset { get; set; } = string.Empty;
	public double Confidence { get; set; }
}

public class DanClassifier : IDisposable
{
	private InferenceSession? _session;
	private string? _inputName;

	public static readonly string[] DisplayNames =
	{
		"Dan 01", "Dan 02", "Dan 03", "Dan 04", "Dan 05",
		"Dan 06", "Dan 07", "Dan 08", "Dan 09", "Dan 10",
		"Dan 11 - α", "Dan 12 - β", "Dan 13 - γ", "Dan 14 - δ", "Dan 15 - ε",
		"Dan 16 - ζ", "Dan 17 - η", "Dan 18 - θ", "Dan 19 - ι", "Dan 20 - κ"
	};

	public static readonly string[] SimpleNames =
	{
		"Dan 1", "Dan 2", "Dan 3", "Dan 4", "Dan 5",
		"Dan 6", "Dan 7", "Dan 8", "Dan 9", "Dan 10",
		"Alpha", "Beta", "Gamma", "Delta", "Epsilon",
		"Zeta", "Eta", "Theta", "Iota", "Kappa"
	};

	public DanClassifier(string? modelPath = null)
	{
		if (string.IsNullOrEmpty(modelPath))
		{
			var baseDir = AppDomain.CurrentDomain.BaseDirectory;
			var c1 = Path.Combine(baseDir, "dan_model.onnx");
			var c2 = Path.Combine(Directory.GetCurrentDirectory(), "dan_model.onnx");
			var c3 = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "dan_model.onnx"));

			if (File.Exists(c1)) modelPath = c1;
			else if (File.Exists(c2)) modelPath = c2;
			else if (File.Exists(c3)) modelPath = c3;
			else throw new FileNotFoundException("dan_model.onnx not found.");
		}

		var options = new SessionOptions();
		options.AppendExecutionProvider_CPU(0);
		_session = new InferenceSession(modelPath, options);
		_inputName = _session.InputMetadata.Keys.First();
	}

	public DanClassification? Classify(MsdScores msd, double interlude, double sunny, bool zeroPadded = true)
	{
		if (_session == null || _inputName == null) return null;

		var features = new float[10];
		features[0] = msd.Overall;
		features[1] = msd.Stream;
		features[2] = msd.Jumpstream;
		features[3] = msd.Handstream;
		features[4] = msd.Stamina;
		features[5] = msd.Jackspeed;
		features[6] = msd.Chordjack;
		features[7] = msd.Technical;
		features[8] = (float)interlude;
		features[9] = (float)sunny;

		var tensor = new DenseTensor<float>(features, new[] { 1, 10 });
		var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

		using var results = _session.Run(inputs);
		var first = results[0];
		float rawValue;

		if (first.Value is float[] floatArray && floatArray.Length > 0) rawValue = floatArray[0];
		else if (first.Value is double[] doubleArray && doubleArray.Length > 0) rawValue = (float)doubleArray[0];
		else rawValue = first.AsTensor<float>().GetValue(0);

		rawValue = Math.Clamp(rawValue, 1.0f, 20.0f) + 1.0f;
		var danLevel = Math.Clamp((int)Math.Round(rawValue), 1, 20);
		var danIndex = danLevel - 1;

		var danName = zeroPadded ? DisplayNames[danIndex] : SimpleNames[danIndex];
		var offset = Math.Abs(rawValue - danLevel);
		var confidence = Math.Max(0.0, 1.0 - offset * 2.0);

		return new DanClassification
		{
			DanIndex = danIndex,
			DanName = danName,
			DominantSkillset = msd.DominantSkillset,
			Confidence = confidence
		};
	}

	public void Dispose()
	{
		_session?.Dispose();
		_session = null;
		GC.SuppressFinalize(this);
	}
}
