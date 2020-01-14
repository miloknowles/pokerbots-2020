#include <boost/python.hpp>
// #include <boost/python/suite/indexing/vector_indexing_suite.hpp>
// #include <boost/python/register_ptr_to_python.hpp>
// #include <boost/shared_ptr.hpp>

#include "permutation_filter.hpp"

using namespace boost::python;
using namespace pb;

BOOST_PYTHON_MODULE(pokerbots_cpp_python)
{
  Py_Initialize();

  class_<ShowdownResult>("ShowdownResult", init<std::string, std::string, std::string>());

  class_<PermutationFilter, boost::noncopyable>("PermutationFilter", init<int>())
    .def("Nonzero", &PermutationFilter::Nonzero)
    .def("Update", &PermutationFilter::Update);
}
