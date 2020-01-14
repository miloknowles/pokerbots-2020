#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
// #include <boost/python/register_ptr_to_python.hpp>
// #include <boost/shared_ptr.hpp>

#include "permutation_filter.hpp"

using namespace boost::python;
using namespace pb;

typedef std::vector<uint8_t> ValueList;

static Permutation ConvertValueList(const ValueList& vl) {
  assert(vl.size() == 13);
  Permutation out;
  for (int i = 0; i < 13; ++i) {
    out[i] = vl[i];
  }
  return out;
}

class PyPermutationFilter : public PermutationFilter {
 public:
  PyPermutationFilter(int N) : PermutationFilter(N) {}

  bool PyHasPermutation(const ValueList& vl) const {
    return HasPermutation(ConvertValueList(vl));
  }
};

BOOST_PYTHON_MODULE(pokerbots_cpp_python)
{
  Py_Initialize();

  class_<ShowdownResult>("ShowdownResult", init<std::string, std::string, std::string>());

  class_<ValueList>("ValueList")
    .def(vector_indexing_suite<ValueList>());

  class_<PyPermutationFilter, boost::noncopyable>("PermutationFilter", init<int>())
    .def("Nonzero", &PyPermutationFilter::Nonzero)
    .def("Update", &PyPermutationFilter::Update)
    .def("HasPermutation", &PyPermutationFilter::PyHasPermutation);
}
