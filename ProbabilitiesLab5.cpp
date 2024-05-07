#include <iostream>
#include <ranges>
#include <filesystem>
#include <fstream>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>

#include <nlohmann/json.hpp>


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, std::pair<T, T>>
T sampleSize(Range values) {
	T count = 0;
	for (const auto& [value, amount] : values) count += amount;
	return count;
}


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, std::pair<T, T>>
T sampleMean(Range values) {
	T mean = 0, count = 0;
	for (const auto& [value, amount] : values) {
		count += amount;
		mean += amount * (value - mean) / count;
	}
	return mean;
}


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, std::pair<T, T>>
T biasedSampleVariance(Range values) {
	T mean = sampleMean<T>(values);
	return sampleMean<T>(values |
		std::views::transform([mean](std::pair<T, T> value) { return std::pair<T, T>(std::pow(value.first - mean, 2), value.second); })
	);
}


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, std::pair<T, T>>
T unbiasedSampleVariance(Range values) {
	T size = sampleSize<T>(values);
	return biasedSampleVariance<T>(values) * size / (size - 1);
}


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, std::pair<T, T>>
T biasedSampleStandardDeviation(Range values) {
	return std::sqrt(biasedSampleVariance<T>(values));
}


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, std::pair<T, T>>
T unbiasedSampleStandardDeviation(Range values) {
	return std::sqrt(unbiasedSampleVariance<T>(values));
}


//template<std::floating_point T>
//T studentsCoefficient(T sampleSize, T confidence) {
//	return boost::math::quantile(
//		boost::math::complement(boost::math::students_t_distribution<T>(sampleSize - 1), (1 - confidence) / 2)
//	);
//}

using nlohmann::json;

json loadSample() {
	auto samplesPath = std::filesystem::path("samples");

	std::vector<std::filesystem::path> sampleFiles;
	std::ranges::copy(std::filesystem::directory_iterator(samplesPath) |
		std::views::transform([](auto directoryEntry) { return directoryEntry.path(); }), std::back_inserter(sampleFiles));

	std::cout << "Available samples:\n";
	for (int sampleIndex = 0; sampleIndex < sampleFiles.size(); sampleIndex++) {
		std::cout << std::format("[{}] {}\n", sampleIndex + 1, sampleFiles[sampleIndex].stem().string());
	}

	int sampleIndex = -1;
	while (!(0 <= sampleIndex && sampleIndex < sampleFiles.size())) {
		std::cout << "Choose sample: ";
		std::cin >> sampleIndex;
		sampleIndex--;
	}

	return json::parse(std::ifstream(sampleFiles[sampleIndex]));
}


using FloatType = double;



std::pair<FloatType, FloatType> meanConfidenceIntervalWithKnownVariance(
	FloatType sampleSize, FloatType statMean, FloatType variance, FloatType confidence
) {
	auto quantile = boost::math::quantile(boost::math::normal(), (confidence + 1) / 2);;
	auto epsilon = std::sqrt(variance / sampleSize) * quantile;
	return { statMean - epsilon, statMean + epsilon };
}

std::pair<FloatType, FloatType> meanConfidenceIntervalWithUnknownVariance(
	FloatType sampleSize, FloatType statMean, FloatType statVariance, FloatType confidence
) {
	auto quantile = boost::math::quantile(boost::math::students_t(sampleSize - 1), (confidence + 1) / 2);;
	auto epsilon = std::sqrt(statVariance / sampleSize) * quantile;
	return { statMean - epsilon, statMean + epsilon };
}


template<std::floating_point T, std::ranges::sized_range Range>
	requires std::is_convertible_v<std::ranges::range_value_t<Range>, T>
inline auto makeVarSeries(Range&& values) {
	return values | std::views::transform([](T value) -> std::pair<T, T> { return { value, 1 }; });
}


const std::vector<std::pair<std::string, std::string>> paramsNames{
	{ "sampleSize", "Sample size" },
	{ "mean", "Mean" },
	{ "variance", "Variance" },
	{ "standardDeviation", "Standard deviation" }
};

const std::vector<std::pair<std::string, std::string>> statisticsNames{
	{ "mean", "Mean" },
	{ "biasedVariance", "Biased variance" },
	{ "unbiasedVariance", "Unbiased variance" },
	{ "biasedStandardDeviation", "Biased standard deviation" },
	{ "unbiasedStandardDeviation", "Unbiased standard deviation" },
};


void calculateStatistics(json& sample, const auto& varSeries) {
	sample["statistics"]["mean"] = sampleMean<FloatType>(varSeries);
	sample["statistics"]["biasedVariance"] = biasedSampleVariance<FloatType>(varSeries);
	sample["statistics"]["unbiasedVariance"] = unbiasedSampleVariance<FloatType>(varSeries);
	sample["statistics"]["biasedStandardDeviation"] = biasedSampleStandardDeviation<FloatType>(varSeries);
	sample["statistics"]["unbiasedStandardDeviation"] = unbiasedSampleStandardDeviation<FloatType>(varSeries);

	sample["params"]["sampleSize"] = sampleSize<FloatType>(varSeries);
}

void calculateStatistics(json& sample) {
	if (sample.contains("values")) {
		auto&& varSeries = makeVarSeries<FloatType>(sample["values"]);
		calculateStatistics(sample, varSeries);
	} else if(sample.contains("variationalSeries")) {
		std::map<std::string, FloatType> keysValues = sample["variationalSeries"].get<std::map<std::string, FloatType>>();
		auto varSeries = keysValues | std::views::transform(
			[](auto keyValue) -> std::pair<FloatType, FloatType> { return { std::stod(keyValue.first), keyValue.second}; }
		);
		calculateStatistics(sample, varSeries);
	}
}


void printParam(const std::string& name, FloatType value) {
	std::cout << std::format("{}: {:.8f}\n", name, value);
}

int main()
{
	auto sample = loadSample();
	calculateStatistics(sample);

	std::cout << "Known parameters:\n";
	for (const auto& [param, name] : paramsNames) {
		if (!sample.contains("params")) break;
		if (sample["params"].contains(param)) {
			printParam(name, sample["params"][param].get<FloatType>());
		}
	}

	std::cout << "\nKnown statistics:\n";
	for (const auto& [statistic, name] : statisticsNames) {
		if (!sample.contains("statistics")) break;
		if (sample["statistics"].contains(statistic)) {
			printParam(name, sample["statistics"][statistic].get<FloatType>());
		}
	}
	std::cout << "\n\n";


	if (sample["meanConfidenceIntervalWithKnownVariance"].get<bool>()) {
		FloatType sampleSize = sample["params"]["sampleSize"];
		FloatType statMean = sample["statistics"]["mean"];
		FloatType variance = sample["params"]["variance"];
		FloatType confidence = sample["confidence"];

		auto interval = meanConfidenceIntervalWithKnownVariance(sampleSize, statMean, variance, confidence);
		std::cout << std::format("Mean confidence interval (with known variance): ({:.8f}, {:.8f}), confidence = {:.2f}",
			interval.first, interval.second, confidence);
	}

	if (sample["meanConfidenceIntervalWithUnknownVariance"].get<bool>()) {
		FloatType sampleSize = sample["params"]["sampleSize"];
		FloatType statMean = sample["statistics"]["mean"];
		FloatType statBiasedVariance = sample["statistics"]["biasedVariance"];
		FloatType confidence = sample["confidence"];

		auto interval = meanConfidenceIntervalWithKnownVariance(sampleSize, statMean, statBiasedVariance, confidence);
		std::cout << std::format("Mean confidence interval (with unknown variance): ({:.8f}, {:.8f}), confidence = {:.2f}",
			interval.first, interval.second, confidence);
	}



	return 0;
}
