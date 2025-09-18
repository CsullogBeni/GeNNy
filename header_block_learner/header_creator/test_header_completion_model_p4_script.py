import unittest
import os


class TestHeaderCompletionModelP4Script(unittest.TestCase):
    """
    Tests for ensuring that the generated P4 scripts contains the correct header blocks extensions.
    """

    def setUp(self) -> None:
        """
        Sets the path of the validation files.
        """
        self.validation_dir = os.path.join(os.path.dirname(__file__), "generated_p4_files")

    def test_basic_p4(self):
        """
        Tests for basic_p4.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "basic_p4.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            " header ethernet_t\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n macAddr_t " +
            "dstAddr;\n macAddr_t srcAddr;\n bit \< 16 \> etherType;\n}",
            content)

    def test_basic_p4_2(self):
        """
        Tests for basic_p4_2.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "basic_p4_2.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            " header ethernet_t\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n macAddr_t " +
            "dstAddr;\n macAddr_t srcAddr;\n bit \< 16 \> etherType;\n}",
            content)

    def test_basic_p4_4(self):
        """
        Tests for basic_p4_4.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "basic_p4_4.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            " header ethernet_t\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n macAddr_t " +
            "dstAddr;\n macAddr_t srcAddr;\n bit \< 16 \> etherType;\n}",
            content)

    def test_basic_p4_8(self):
        """
        Tests for basic_p4_8.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "basic_p4_8.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            " header ethernet_t\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n macAddr_t " +
            "dstAddr;\n macAddr_t srcAddr;\n bit \< 16 \> etherType;\n}",
            content)

    def test_basic_p4_16(self):
        """
        Tests for basic_p4_16.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "basic_p4_16.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            " header ethernet_t macAddr_t dstAddr;\n macAddr_t destinationAddress;\n macAddr_t " +
            "sourceAddress;\n macAddr_t srcAddr;\n bit 16 \< \> etherType;\n\{\n}",
            content)

    def test_basic_p4_with_new_header_validation_0(self):
        """
        Tests for basic_p4_with_new_header_validation_0.p4 file, whether it contains the correct header blocks
        extensions.
        """
        path = os.path.join(self.validation_dir, "basic_p4_with_new_header_validation_0.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            "header my_header\\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n}\n",
            content)

    def test_ex1(self):
        """
        Tests for ex1.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "ex1.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn(
            'header ethernet_t\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n',
            content)

    def test_fabric(self):
        """
        Tests for fabric.p4 file, whether it contains the correct header blocks extensions.
        """
        path = os.path.join(self.validation_dir, "fabric.p4")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertNotIn(
            'header ethernet_t\{\n macAddr_t destinationAddress;\n macAddr_t sourceAddress;\n',
            content)


if __name__ == '__main__':
    unittest.main()
