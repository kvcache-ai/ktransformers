#include <iostream>
#include <iterator>
#include <vector>

int main() {
  std::vector<int> v = {0, 1, 2, 3, 4, 5};

  using RevIt = std::reverse_iterator<std::vector<int>::iterator>;

  const auto it = v.begin() + 3;
  RevIt r_it{it};

  std::cout << "*it == " << *it << '\n'
            << "*r_it == " << *r_it << '\n'
            << "*r_it.base() == " << *r_it.base() << '\n'
            << "*(r_it.base()-1) == " << *(r_it.base() - 1) << '\n';

  RevIt r_end{v.begin()};
  RevIt r_begin{v.end()};

  for (auto it = r_end.base(); it != r_begin.base(); ++it)
    std::cout << *it << ' ';
  std::cout << '\n';

  for (auto it = r_begin; it != r_end; ++it)
    std::cout << *it << ' ';
  std::cout << '\n';

  for (auto it = r_begin; it != r_end; ++it) {
    if (*it == 3) {
      v.erase(std::next(it).base());
    }
  }

  for (auto it : v)
    std::cout << it << ' ';
  std::cout << '\n';
}