/**
 * 🧬 Genomics Laboratory Interface
 * 
 * Comprehensive interface for eDNA processing, taxonomic classification, 
 * diversity analysis, and phylogenetic reconstruction.
 * 
 * @author Ocean-Bio Development Team
 * @version 2.0.0
 */

import React, { useState, useCallback, useMemo } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Upload, 
  Select, 
  Table, 
  Tabs, 
  Progress, 
  Space, 
  Typography, 
  Statistic, 
  Tree, 
  Tag,
  Alert,
  Spin,
  Modal,
  Form,
  Input,
  Slider,
  Collapse,
  Divider,
  Steps,
  Result
} from 'antd';
import {
  InboxOutlined,
  DNAOutlined,
  BranchesOutlined,
  PieChartOutlined,
  ExperimentOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  StopOutlined,
  EyeOutlined,
  DeleteOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  LineChart,
  Line,
  Legend,
  TreeMap
} from 'recharts';
import { useQuery, useMutation } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';

// API imports
import { genomicsAPI } from '../../services/api';

// Types
interface SequenceData {
  id: string;
  sequence: string;
  quality_score?: number;
  length: number;
  gc_content?: number;
  processed: boolean;
  classification?: {
    taxonomy: Record<string, string>;
    confidence: Record<string, number>;
    method: string;
  };
}

interface DiversityMetrics {
  shannon: number;
  simpson: number;
  chao1: number;
  observed_species: number;
  pielou_evenness: number;
}

interface PhylogeneticTree {
  newick: string;
  nodes: Array<{
    name: string;
    branch_length: number;
    bootstrap?: number;
  }>;
  statistics: {
    total_length: number;
    height: number;
    nodes: number;
  };
}

const { Option } = Select;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Dragger } = Upload;
const { Step } = Steps;
const { Panel } = Collapse;

interface GenomicsLabProps {
  className?: string;
}

const GenomicsLab: React.FC<GenomicsLabProps> = ({ className = '' }) => {
  // State management
  const [sequences, setSequences] = useState<SequenceData[]>([]);
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedSequences, setSelectedSequences] = useState<string[]>([]);
  const [processingStep, setProcessingStep] = useState(0);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [treeViewModal, setTreeViewModal] = useState(false);
  const [classificationSettings, setClassificationSettings] = useState({
    method: 'consensus',
    confidence_threshold: 0.7,
    database: 'marine_db'
  });

  // API calls
  const sequenceProcessingMutation = useMutation({
    mutationFn: (data: any) => genomicsAPI.processSequences(data),
    onSuccess: (data) => {
      toast.success('Sequences processed successfully');
      setProcessingStep(1);
    },
    onError: (error) => {
      console.error('Processing failed:', error);
      toast.error('Sequence processing failed');
    }
  });

  const classificationMutation = useMutation({
    mutationFn: (data: any) => genomicsAPI.classifySequences(data),
    onSuccess: (data) => {
      toast.success('Taxonomic classification completed');
      setProcessingStep(2);
      // Update sequences with classification results
      updateSequencesWithClassification(data.assignments);
    },
    onError: (error) => {
      console.error('Classification failed:', error);
      toast.error('Taxonomic classification failed');
    }
  });

  const diversityMutation = useMutation({
    mutationFn: (data: any) => genomicsAPI.analyzeDiversity(data),
    onSuccess: (data) => {
      toast.success('Diversity analysis completed');
      setAnalysisResults(prev => ({ ...prev, diversity: data.results }));
    },
    onError: (error) => {
      console.error('Diversity analysis failed:', error);
      toast.error('Diversity analysis failed');
    }
  });

  const phylogeneticMutation = useMutation({
    mutationFn: (data: any) => genomicsAPI.analyzePhylogenetics(data),
    onSuccess: (data) => {
      toast.success('Phylogenetic analysis completed');
      setAnalysisResults(prev => ({ ...prev, phylogenetics: data.tree_results }));
      setProcessingStep(3);
    },
    onError: (error) => {
      console.error('Phylogenetic analysis failed:', error);
      toast.error('Phylogenetic analysis failed');
    }
  });

  // Mock data
  const mockSequences: SequenceData[] = useMemo(() => [
    {
      id: 'seq_001',
      sequence: 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC',
      quality_score: 0.92,
      length: 47,
      gc_content: 0.51,
      processed: true,
      classification: {
        taxonomy: {
          kingdom: 'Animalia',
          phylum: 'Chordata',
          class: 'Actinopterygii',
          family: 'Scombridae',
          genus: 'Thunnus',
          species: 'Thunnus albacares'
        },
        confidence: {
          kingdom: 0.98,
          phylum: 0.95,
          class: 0.91,
          family: 0.85,
          genus: 0.78,
          species: 0.72
        },
        method: 'consensus'
      }
    },
    {
      id: 'seq_002',
      sequence: 'GCTATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGC',
      quality_score: 0.88,
      length: 47,
      gc_content: 0.53,
      processed: true,
      classification: {
        taxonomy: {
          kingdom: 'Animalia',
          phylum: 'Cnidaria',
          class: 'Anthozoa',
          family: 'Acroporidae',
          genus: 'Acropora',
          species: 'Acropora cervicornis'
        },
        confidence: {
          kingdom: 0.99,
          phylum: 0.94,
          class: 0.89,
          family: 0.81,
          genus: 0.75,
          species: 0.68
        },
        method: 'consensus'
      }
    }
  ], []);

  const mockDiversityData = useMemo(() => ({
    shannon: 2.45,
    simpson: 0.85,
    chao1: 15.2,
    observed_species: 12,
    pielou_evenness: 0.78,
    rarefaction_points: Array.from({ length: 20 }, (_, i) => ({
      sample_size: (i + 1) * 5,
      species_count: Math.min(12, Math.log(i + 1) * 4 + Math.random() * 2)
    }))
  }), []);

  // Handlers
  const handleFileUpload = useCallback((info: any) => {
    const { status } = info.file;
    
    if (status === 'done') {
      // Parse FASTA file (mock)
      const newSequences = Array.from({ length: 5 }, (_, i) => ({
        id: `uploaded_${i + 1}`,
        sequence: 'ATGC'.repeat(10 + Math.random() * 20),
        length: 40 + Math.random() * 40,
        gc_content: 0.4 + Math.random() * 0.2,
        processed: false
      }));
      
      setSequences(prev => [...prev, ...newSequences]);
      toast.success(`${info.file.name} uploaded successfully`);
    } else if (status === 'error') {
      toast.error(`${info.file.name} upload failed`);
    }
  }, []);

  const handleProcessSequences = useCallback(() => {
    const sequenceData = sequences
      .filter(seq => selectedSequences.includes(seq.id))
      .reduce((acc, seq) => {
        acc[seq.id] = seq.sequence;
        return acc;
      }, {} as Record<string, string>);

    sequenceProcessingMutation.mutate({
      sequences: sequenceData,
      quality_threshold: 0.8,
      min_length: 50,
      remove_primers: true,
      trim_low_quality: true
    });
  }, [sequences, selectedSequences, sequenceProcessingMutation]);

  const handleClassifySequences = useCallback(() => {
    const sequenceData = sequences
      .filter(seq => selectedSequences.includes(seq.id))
      .reduce((acc, seq) => {
        acc[seq.id] = seq.sequence;
        return acc;
      }, {} as Record<string, string>);

    classificationMutation.mutate({
      sequences: sequenceData,
      method: classificationSettings.method,
      min_confidence: classificationSettings.confidence_threshold,
      database: classificationSettings.database
    });
  }, [sequences, selectedSequences, classificationSettings, classificationMutation]);

  const handleDiversityAnalysis = useCallback(() => {
    // Build abundance data from classifications
    const abundanceData: Record<string, number> = {};
    
    sequences
      .filter(seq => seq.classification)
      .forEach(seq => {
        const species = seq.classification?.taxonomy.species;
        if (species) {
          abundanceData[species] = (abundanceData[species] || 0) + 1;
        }
      });

    diversityMutation.mutate({
      abundance_data: abundanceData,
      sample_id: 'sample_001',
      analysis_type: 'alpha'
    });
  }, [sequences, diversityMutation]);

  const handlePhylogeneticAnalysis = useCallback(() => {
    const sequenceData = sequences
      .filter(seq => selectedSequences.includes(seq.id))
      .reduce((acc, seq) => {
        acc[seq.id] = seq.sequence;
        return acc;
      }, {} as Record<string, string>);

    phylogeneticMutation.mutate({
      sequences: sequenceData,
      distance_method: 'kimura_2p',
      tree_method: 'neighbor_joining',
      bootstrap_replicates: 100
    });
  }, [sequences, selectedSequences, phylogeneticMutation]);

  const updateSequencesWithClassification = useCallback((assignments: any) => {
    setSequences(prev => prev.map(seq => {
      const assignment = assignments[seq.id];
      if (assignment) {
        return {
          ...seq,
          classification: {
            taxonomy: assignment.taxonomy,
            confidence: assignment.confidence_scores,
            method: assignment.method_used
          }
        };
      }
      return seq;
    }));
  }, []);

  // Table columns
  const sequenceColumns = [
    {
      title: 'Sequence ID',
      dataIndex: 'id',
      key: 'id',
      render: (text: string) => <Text code>{text}</Text>
    },
    {
      title: 'Length',
      dataIndex: 'length',
      key: 'length',
      render: (length: number) => `${length} bp`
    },
    {
      title: 'GC Content',
      dataIndex: 'gc_content',
      key: 'gc_content',
      render: (gc: number) => gc ? `${(gc * 100).toFixed(1)}%` : 'N/A'
    },
    {
      title: 'Quality',
      dataIndex: 'quality_score',
      key: 'quality_score',
      render: (score: number) => (
        <Progress 
          percent={score ? Math.round(score * 100) : 0} 
          size="small"
          status={score > 0.8 ? 'success' : score > 0.6 ? 'active' : 'exception'}
        />
      )
    },
    {
      title: 'Classification',
      key: 'classification',
      render: (record: SequenceData) => {
        if (record.classification) {
          const species = record.classification.taxonomy.species;
          const confidence = record.classification.confidence.species;
          return (
            <Space direction="vertical" size="small">
              <Text strong>{species}</Text>
              <Tag color={confidence > 0.8 ? 'green' : confidence > 0.6 ? 'orange' : 'red'}>
                {(confidence * 100).toFixed(0)}% confidence
              </Tag>
            </Space>
          );
        }
        return <Text type="secondary">Not classified</Text>;
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: SequenceData) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />}>View</Button>
          <Button size="small" icon={<DeleteOutlined />} danger>Remove</Button>
        </Space>
      )
    }
  ];

  // Charts
  const TaxonomicPieChart = () => {
    const data = Object.entries(
      sequences
        .filter(seq => seq.classification)
        .reduce((acc, seq) => {
          const phylum = seq.classification?.taxonomy.phylum || 'Unknown';
          acc[phylum] = (acc[phylum] || 0) + 1;
          return acc;
        }, {} as Record<string, number>)
    ).map(([name, value]) => ({ name, value }));

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

    return (
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <RechartsTooltip />
        </PieChart>
      </ResponsiveContainer>
    );
  };

  const RarefactionChart = () => (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={mockDiversityData.rarefaction_points}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="sample_size" />
        <YAxis />
        <RechartsTooltip />
        <Line type="monotone" dataKey="species_count" stroke="#8884d8" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderProcessingSteps = () => (
    <Steps current={processingStep} className="mb-4">
      <Step title="Upload & Process" description="Quality control and filtering" />
      <Step title="Classify" description="Taxonomic identification" />
      <Step title="Analyze" description="Diversity and phylogenetics" />
      <Step title="Results" description="Comprehensive analysis" />
    </Steps>
  );

  return (
    <div className={`genomics-lab ${className}`}>
      {/* Header */}
      <Card className="lab-header mb-4">
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              🧬 Genomics Laboratory
            </Title>
            <Text type="secondary">
              Environmental DNA analysis, taxonomic classification, and phylogenetic reconstruction
            </Text>
          </Col>
          <Col>
            <Space>
              <Button icon={<ReloadOutlined />}>Reset</Button>
              <Button icon={<DownloadOutlined />}>Export Results</Button>
            </Space>
          </Col>
        </Row>
        
        <Divider />
        {renderProcessingSteps()}
      </Card>

      {/* Main Content */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        {/* Upload & Processing Tab */}
        <TabPane tab="📁 Upload & Process" key="upload">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="Sequence Upload">
                <Dragger
                  name="file"
                  multiple
                  accept=".fasta,.fa,.fastq,.fq"
                  onChange={handleFileUpload}
                  showUploadList={false}
                >
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                  </p>
                  <p className="ant-upload-text">
                    Click or drag FASTA/FASTQ files to upload
                  </p>
                  <p className="ant-upload-hint">
                    Support for single or multiple file upload. Accepted formats: .fasta, .fastq
                  </p>
                </Dragger>
              </Card>

              <Card title="Sequence Data" className="mt-4">
                <Table
                  dataSource={[...mockSequences, ...sequences]}
                  columns={sequenceColumns}
                  rowKey="id"
                  rowSelection={{
                    selectedRowKeys: selectedSequences,
                    onChange: (keys) => setSelectedSequences(keys as string[])
                  }}
                  size="small"
                  pagination={{ pageSize: 10 }}
                />
              </Card>
            </Col>

            <Col span={8}>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Card size="small">
                  <Statistic
                    title="Total Sequences"
                    value={mockSequences.length + sequences.length}
                    prefix={<DNAOutlined />}
                  />
                </Card>

                <Card size="small">
                  <Statistic
                    title="Selected"
                    value={selectedSequences.length}
                    suffix={`/ ${mockSequences.length + sequences.length}`}
                  />
                </Card>

                <Card size="small">
                  <Statistic
                    title="Quality Passed"
                    value={mockSequences.filter(s => (s.quality_score || 0) > 0.8).length}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>

                <Card title="Processing Options" size="small">
                  <Form layout="vertical">
                    <Form.Item label="Quality Threshold">
                      <Slider min={0.5} max={1.0} step={0.1} defaultValue={0.8} />
                    </Form.Item>
                    
                    <Form.Item label="Minimum Length">
                      <Slider min={20} max={200} defaultValue={50} />
                    </Form.Item>
                    
                    <Button 
                      type="primary" 
                      block
                      icon={<PlayCircleOutlined />}
                      onClick={handleProcessSequences}
                      loading={sequenceProcessingMutation.isLoading}
                      disabled={selectedSequences.length === 0}
                    >
                      Process Sequences
                    </Button>
                  </Form>
                </Card>
              </Space>
            </Col>
          </Row>
        </TabPane>

        {/* Classification Tab */}
        <TabPane tab="🔬 Taxonomic Classification" key="classification">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="Classification Results">
                {mockSequences.filter(s => s.classification).length > 0 ? (
                  <TaxonomicPieChart />
                ) : (
                  <Result
                    icon={<ExperimentOutlined />}
                    title="No Classification Results"
                    subTitle="Run taxonomic classification to see results here"
                    extra={
                      <Button 
                        type="primary" 
                        onClick={handleClassifySequences}
                        disabled={selectedSequences.length === 0}
                      >
                        Start Classification
                      </Button>
                    }
                  />
                )}
              </Card>

              <Card title="Taxonomic Composition" className="mt-4">
                <Collapse>
                  {['Kingdom', 'Phylum', 'Class', 'Family', 'Genus', 'Species'].map(rank => (
                    <Panel key={rank} header={`${rank} Level`}>
                      <Table
                        dataSource={Object.entries(
                          mockSequences
                            .filter(s => s.classification)
                            .reduce((acc, seq) => {
                              const taxon = seq.classification?.taxonomy[rank.toLowerCase()] || 'Unknown';
                              if (!acc[taxon]) acc[taxon] = { taxon, count: 0, avg_confidence: 0 };
                              acc[taxon].count++;
                              return acc;
                            }, {} as Record<string, any>)
                        ).map(([_, data]) => data)}
                        columns={[
                          { title: 'Taxon', dataIndex: 'taxon', key: 'taxon' },
                          { title: 'Count', dataIndex: 'count', key: 'count' },
                          { 
                            title: 'Confidence', 
                            dataIndex: 'avg_confidence', 
                            key: 'confidence',
                            render: (conf: number) => `${(conf * 100).toFixed(1)}%`
                          }
                        ]}
                        size="small"
                        pagination={false}
                      />
                    </Panel>
                  ))}
                </Collapse>
              </Card>
            </Col>

            <Col span={8}>
              <Card title="Classification Settings" size="small">
                <Form layout="vertical">
                  <Form.Item label="Method">
                    <Select 
                      value={classificationSettings.method}
                      onChange={(value) => setClassificationSettings(prev => ({ ...prev, method: value }))}
                    >
                      <Option value="consensus">Consensus</Option>
                      <Option value="blast_like">BLAST-like</Option>
                      <Option value="kmer_based">K-mer Based</Option>
                      <Option value="naive_bayes">Naive Bayes</Option>
                    </Select>
                  </Form.Item>
                  
                  <Form.Item label="Confidence Threshold">
                    <Slider 
                      min={0.5} 
                      max={1.0} 
                      step={0.1} 
                      value={classificationSettings.confidence_threshold}
                      onChange={(value) => setClassificationSettings(prev => ({ ...prev, confidence_threshold: value }))}
                    />
                  </Form.Item>
                  
                  <Form.Item label="Reference Database">
                    <Select 
                      value={classificationSettings.database}
                      onChange={(value) => setClassificationSettings(prev => ({ ...prev, database: value }))}
                    >
                      <Option value="marine_db">Marine Species DB</Option>
                      <Option value="ncbi">NCBI GenBank</Option>
                      <Option value="bold">BOLD Systems</Option>
                    </Select>
                  </Form.Item>
                  
                  <Button 
                    type="primary" 
                    block
                    icon={<PlayCircleOutlined />}
                    onClick={handleClassifySequences}
                    loading={classificationMutation.isLoading}
                    disabled={selectedSequences.length === 0}
                  >
                    Run Classification
                  </Button>
                </Form>
              </Card>

              <Card title="Classification Summary" size="small" className="mt-4">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Statistic
                    title="Classified Sequences"
                    value={mockSequences.filter(s => s.classification).length}
                    suffix={`/ ${mockSequences.length}`}
                  />
                  
                  <Statistic
                    title="Unique Species"
                    value={new Set(mockSequences
                      .filter(s => s.classification)
                      .map(s => s.classification?.taxonomy.species)
                    ).size}
                  />
                  
                  <Statistic
                    title="Average Confidence"
                    value={85}
                    suffix="%"
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* Diversity Analysis Tab */}
        <TabPane tab="📊 Diversity Analysis" key="diversity">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="Rarefaction Curve">
                <RarefactionChart />
              </Card>
            </Col>

            <Col span={8}>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Card size="small">
                  <Statistic
                    title="Shannon Diversity"
                    value={mockDiversityData.shannon}
                    precision={3}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>

                <Card size="small">
                  <Statistic
                    title="Simpson Index"
                    value={mockDiversityData.simpson}
                    precision={3}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>

                <Card size="small">
                  <Statistic
                    title="Chao1 Estimator"
                    value={mockDiversityData.chao1}
                    precision={1}
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Card>

                <Card size="small">
                  <Statistic
                    title="Pielou's Evenness"
                    value={mockDiversityData.pielou_evenness}
                    precision={3}
                    valueStyle={{ color: '#fa8c16' }}
                  />
                </Card>

                <Button 
                  type="primary" 
                  block
                  icon={<PieChartOutlined />}
                  onClick={handleDiversityAnalysis}
                  loading={diversityMutation.isLoading}
                >
                  Calculate Diversity
                </Button>
              </Space>
            </Col>
          </Row>
        </TabPane>

        {/* Phylogenetics Tab */}
        <TabPane tab="🌳 Phylogenetics" key="phylogenetics">
          <Row gutter={16}>
            <Col span={16}>
              <Card 
                title="Phylogenetic Tree" 
                extra={
                  <Button 
                    icon={<EyeOutlined />} 
                    onClick={() => setTreeViewModal(true)}
                  >
                    View Full Tree
                  </Button>
                }
              >
                <div className="tree-container">
                  <Text type="secondary">
                    Phylogenetic tree visualization will appear here after analysis
                  </Text>
                  <br />
                  <Button 
                    type="primary"
                    icon={<BranchesOutlined />}
                    onClick={handlePhylogeneticAnalysis}
                    loading={phylogeneticMutation.isLoading}
                    disabled={selectedSequences.length < 3}
                  >
                    Construct Tree
                  </Button>
                </div>
              </Card>
            </Col>

            <Col span={8}>
              <Card title="Tree Statistics" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Statistic title="Total Branch Length" value={2.45} precision={3} />
                  <Statistic title="Tree Height" value={0.87} precision={3} />
                  <Statistic title="Internal Nodes" value={8} />
                  <Statistic title="Bootstrap Support" value={92} suffix="%" />
                </Space>
              </Card>

              <Card title="Analysis Settings" size="small" className="mt-4">
                <Form layout="vertical">
                  <Form.Item label="Distance Method">
                    <Select defaultValue="kimura_2p">
                      <Option value="kimura_2p">Kimura 2-Parameter</Option>
                      <Option value="jukes_cantor">Jukes-Cantor</Option>
                      <Option value="p_distance">P-Distance</Option>
                    </Select>
                  </Form.Item>
                  
                  <Form.Item label="Tree Method">
                    <Select defaultValue="neighbor_joining">
                      <Option value="neighbor_joining">Neighbor-Joining</Option>
                      <Option value="upgma">UPGMA</Option>
                      <Option value="maximum_parsimony">Maximum Parsimony</Option>
                    </Select>
                  </Form.Item>
                  
                  <Form.Item label="Bootstrap Replicates">
                    <Slider min={10} max={1000} defaultValue={100} />
                  </Form.Item>
                </Form>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* Tree View Modal */}
      <Modal
        title="Phylogenetic Tree Viewer"
        visible={treeViewModal}
        onCancel={() => setTreeViewModal(false)}
        width={800}
        footer={null}
      >
        <div style={{ height: 500, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Text type="secondary">Interactive tree visualization would appear here</Text>
        </div>
      </Modal>

      <style jsx>{`
        .genomics-lab {
          padding: 0;
        }
        
        .tree-container {
          height: 300px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          border: 2px dashed #d9d9d9;
          border-radius: 6px;
          background: #fafafa;
        }
      `}</style>
    </div>
  );
};

export default GenomicsLab;