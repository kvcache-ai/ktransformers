#ifndef CSV_READER_HPP
#define CSV_READER_HPP

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace csv {

/**
 * @brief Parses a CSV line into individual fields, handling quoted fields with
 * commas and newlines.
 *
 * @param line The CSV line to parse.
 * @return A vector of strings, each representing a field in the CSV line.
 */
inline std::vector<std::string> parse_csv_line(const std::string &line) {
  std::vector<std::string> result;
  std::string field;
  bool in_quotes = false;

  for (size_t i = 0; i < line.length(); ++i) {
    char c = line[i];

    if (c == '"') {
      // Handle double quotes inside quoted fields
      if (in_quotes && i + 1 < line.length() && line[i + 1] == '"') {
        field += '"';
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == ',' && !in_quotes) {
      result.push_back(field);
      field.clear();
    } else {
      field += c;
    }
  }
  result.push_back(field);
  return result;
}

/**
 * @brief Reads a CSV file and returns a vector of pairs containing column names
 * and their corresponding data vectors.
 *
 * This function reads the header to obtain column names and uses multithreading
 * to read and parse the CSV file in chunks.
 *
 * @param filename The path to the CSV file.
 * @return A vector of pairs, each containing a column name and a vector of data
 * for that column.
 */
inline std::vector<std::pair<std::string, std::vector<std::string>>>
read_csv(const std::string &filename) {
  std::cout << "Reading CSV file: " << filename << std::endl;
  // Open the file
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Cannot open file");
  }

  // Read the header line and parse column names
  std::string header_line;
  std::getline(file, header_line);
  std::vector<std::string> column_names = parse_csv_line(header_line);

  // Prepare the result vector with column names
  std::vector<std::pair<std::string, std::vector<std::string>>> result;
  for (const auto &name : column_names) {
    result.emplace_back(name, std::vector<std::string>());
  }

  // Read the rest of the file into a string buffer
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  // Determine the number of threads to use
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4; // Default to 4 threads if hardware_concurrency returns 0

  // Calculate chunk start positions based on content size
  std::vector<size_t> chunk_starts;
  size_t content_size = content.size();
  size_t chunk_size = content_size / num_threads;

  chunk_starts.push_back(0);
  for (unsigned int i = 1; i < num_threads; ++i) {
    size_t pos = i * chunk_size;
    // Adjust position to the next newline character to ensure we start at the
    // beginning of a line
    while (pos < content_size && content[pos] != '\n') {
      ++pos;
    }
    if (pos < content_size) {
      ++pos; // Skip the newline character
    }
    chunk_starts.push_back(pos);
  }
  chunk_starts.push_back(content_size);

  // Create threads to parse each chunk
  std::vector<std::vector<std::vector<std::string>>> thread_results(
      num_threads);
  std::vector<std::thread> threads;

  for (unsigned int i = 0; i < num_threads; ++i) {
    size_t start = chunk_starts[i];
    size_t end = chunk_starts[i + 1];

    threads.emplace_back([&content, start, end, &thread_results, i]() {
      std::vector<std::vector<std::string>> local_result;
      size_t pos = start;
      while (pos < end) {
        size_t next_pos = content.find('\n', pos);
        if (next_pos == std::string::npos || next_pos > end) {
          next_pos = end;
        }
        std::string line = content.substr(pos, next_pos - pos);
        if (!line.empty()) {
          local_result.push_back(parse_csv_line(line));
        }
        pos = next_pos + 1;
      }
      thread_results[i] = std::move(local_result);
    });
  }

  // Wait for all threads to finish
  for (auto &t : threads) {
    t.join();
  }

  // Combine the results from all threads into the final result
  for (const auto &local_result : thread_results) {
    for (const auto &row : local_result) {
      for (size_t i = 0; i < row.size(); ++i) {
        if (i < result.size()) {
          result[i].second.push_back(row[i]);
        }
      }
    }
  }

  return result;
}

/**
 * @brief Writes the CSV data into a file.
 *
 * @param filename The path to the output CSV file.
 * @param data A vector of pairs, each containing a column name and a vector of
 * data for that column.
 */
inline void write_csv(
    const std::string &filename,
    const std::vector<std::pair<std::string, std::vector<std::string>>> &data) {
  std::cout << "Writing CSV file: " << filename << std::endl;

  // Open the file for writing
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing");
  }

  // Check that all columns have the same number of rows
  if (data.empty()) {
    return; // Nothing to write
  }
  size_t num_rows = data[0].second.size();
  for (const auto &column : data) {
    if (column.second.size() != num_rows) {
      throw std::runtime_error("All columns must have the same number of rows");
    }
  }

  // Write the header
  for (size_t i = 0; i < data.size(); ++i) {
    file << data[i].first;
    if (i != data.size() - 1) {
      file << ',';
    }
  }
  file << '\n';

  // Write the data rows
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t col = 0; col < data.size(); ++col) {
      const std::string &field = data[col].second[row];
      // Handle CSV escaping
      std::string escaped_field = field;
      bool needs_quotes = false;
      if (escaped_field.find('"') != std::string::npos) {
        needs_quotes = true;
        // Escape double quotes
        size_t pos = 0;
        while ((pos = escaped_field.find('"', pos)) != std::string::npos) {
          escaped_field.insert(pos, "\"");
          pos += 2;
        }
      }
      if (escaped_field.find(',') != std::string::npos ||
          escaped_field.find('\n') != std::string::npos) {
        needs_quotes = true;
      }
      if (needs_quotes) {
        file << '"' << escaped_field << '"';
      } else {
        file << escaped_field;
      }
      if (col != data.size() - 1) {
        file << ',';
      }
    }
    file << '\n';
  }
}

} // namespace csv

#endif // CSV_READER_HPP
