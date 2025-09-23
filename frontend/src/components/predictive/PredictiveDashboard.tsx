/**
 * 📊 Predictive Analytics Dashboard
 * 
 * Comprehensive dashboard for stock assessment, forecasting, and trend analysis.
 * Features interactive charts, model comparison, and predictive visualizations.
 * 
 * @author Ocean-Bio Development Team
 * @version 2.0.0
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Select, 
  Button, 
  DatePicker, 
  Slider, 
  Space, 
  Typography, 
  Statistic, 
  Alert, 
  Spin, 
  Tabs,
  Table,
  Progress,
  Tag,
  Tooltip,
  Modal,
  Form,
  InputNumber,
  Upload
} from 'antd';
import { 
  LineChartOutlined, 
  BarChartOutlined,
  PieChartOutlined,
  TrendingUpOutlined,
  WarningOutlined,
  DownloadOutlined,
  UploadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  StopOutlined
} from '@ant-design/icons';
import { 
  LineChart, 
  Line, 
  AreaChart,
  Area,
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  Brush,
  ComposedChart
} from 'recharts';
import { useQuery, useMutation } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import dayjs from 'dayjs';

// API imports
import { predictiveAPI } from '../../services/api';

// Types
interface TimeSeriesData {
  date: string;
  value: number;
  predicted?: boolean;
  lower_bound?: number;
  upper_bound?: number;
  category?: string;
}

interface StockAssessmentResult {
  species: string;
  assessment_type: string;
  status: 'healthy' | 'overfished' | 'overfishing' | 'unknown';
  biomass: number;
  fishing_mortality: number;
  spawning_stock: number;
  recruitment: number;
  msy: number; // Maximum Sustainable Yield
  confidence_interval: [number, number];
  recommendations: string[];
  last_updated: string;
}

interface ForecastResult {
  forecast_type: string;
  horizon: number;
  confidence_level: number;
  predictions: TimeSeriesData[];
  accuracy_metrics: {
    mse: number;
    mae: number;
    mape: number;
    r2: number;
  };
  model_info: {
    type: string;
    parameters: Record<string, any>;
    training_period: [string, string];
  };
}

interface TrendAnalysisResult {
  variable: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  trend_strength: number;
  seasonality: boolean;
  change_points: string[];
  correlation_matrix: Record<string, Record<string, number>>;
}

const { Option } = Select;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;

interface PredictiveDashboardProps {
  className?: string;
}

const PredictiveDashboard: React.FC<PredictiveDashboardProps> = ({ className = '' }) => {
  // State management
  const [selectedSpecies, setSelectedSpecies] = useState<string[]>(['all']);
  const [selectedModels, setSelectedModels] = useState<string[]>(['arima']);
  const [forecastHorizon, setForecastHorizon] = useState(12);
  const [confidenceLevel, setConfidenceLevel] = useState(0.95);
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);
  const [activeTab, setActiveTab] = useState('forecasting');
  const [modelConfigModal, setModelConfigModal] = useState(false);
  const [runningAnalysis, setRunningAnalysis] = useState(false);

  // Data fetching
  const {
    data: availableModels,
    isLoading: modelsLoading,
    error: modelsError
  } = useQuery({
    queryKey: ['predictive-models'],
    queryFn: () => predictiveAPI.getAvailableModels(),
    refetchInterval: 60000
  });

  const {
    data: forecastData,
    isLoading: forecastLoading,
    refetch: refetchForecast
  } = useQuery({
    queryKey: ['forecast-data', selectedSpecies, selectedModels, forecastHorizon],
    queryFn: () => predictiveAPI.generateForecast({
      forecast_type: selectedModels[0] || 'arima',
      time_series_data: [], // This would be populated with historical data
      forecast_horizon: forecastHorizon,
      confidence_level: confidenceLevel,
      model_parameters: {}
    }),
    enabled: selectedSpecies.length > 0 && selectedModels.length > 0,
    refetchOnWindowFocus: false
  });

  const {
    data: stockAssessments,
    isLoading: assessmentLoading,
    refetch: refetchAssessments
  } = useQuery({
    queryKey: ['stock-assessments', selectedSpecies],
    queryFn: async () => {
      const results = [];
      for (const species of selectedSpecies) {
        if (species !== 'all') {
          try {
            const assessment = await predictiveAPI.performStockAssessment({
              species_name: species,
              data_source: 'fisheries_db',
              model_type: 'surplus_production',
              parameters: {},
              time_series_data: []
            });
            results.push(assessment);
          } catch (error) {
            console.error(`Assessment failed for ${species}:`, error);
          }
        }
      }
      return results;
    },
    enabled: selectedSpecies.length > 0 && !selectedSpecies.includes('all'),
    refetchOnWindowFocus: false
  });

  // Mutations
  const trainModelMutation = useMutation({
    mutationFn: (modelConfig: any) => predictiveAPI.trainModel(modelConfig),
    onSuccess: () => {
      toast.success('Model training started');
      setModelConfigModal(false);
    },
    onError: (error) => {
      console.error('Model training failed:', error);
      toast.error('Failed to start model training');
    }
  });

  const trendAnalysisMutation = useMutation({
    mutationFn: (analysisConfig: any) => predictiveAPI.analyzeTrends(analysisConfig),
    onSuccess: (data) => {
      toast.success('Trend analysis completed');
    },
    onError: (error) => {
      console.error('Trend analysis failed:', error);
      toast.error('Trend analysis failed');
    }
  });

  // Mock data for demonstration
  const mockForecastData: TimeSeriesData[] = useMemo(() => {
    const baseDate = dayjs().subtract(24, 'month');
    const data: TimeSeriesData[] = [];
    
    // Historical data
    for (let i = 0; i < 24; i++) {
      data.push({
        date: baseDate.add(i, 'month').format('YYYY-MM'),
        value: 100 + Math.sin(i / 3) * 20 + Math.random() * 10,
        predicted: false
      });
    }
    
    // Forecast data
    for (let i = 0; i < forecastHorizon; i++) {
      const forecastValue = 100 + Math.sin((24 + i) / 3) * 20;
      data.push({
        date: baseDate.add(24 + i, 'month').format('YYYY-MM'),
        value: forecastValue,
        predicted: true,
        lower_bound: forecastValue - 15,
        upper_bound: forecastValue + 15
      });
    }
    
    return data;
  }, [forecastHorizon]);

  const mockStockStatus: StockAssessmentResult[] = useMemo(() => [
    {
      species: 'Hilsa shad',
      assessment_type: 'surplus_production',
      status: 'overfished',
      biomass: 45000,
      fishing_mortality: 0.35,
      spawning_stock: 12000,
      recruitment: 8500,
      msy: 15000,
      confidence_interval: [12000, 18000],
      recommendations: [
        'Reduce fishing pressure by 25%',
        'Implement seasonal closures during spawning',
        'Enhance monitoring programs'
      ],
      last_updated: '2024-09-23'
    },
    {
      species: 'Indian mackerel',
      assessment_type: 'virtual_population',
      status: 'healthy',
      biomass: 85000,
      fishing_mortality: 0.18,
      spawning_stock: 32000,
      recruitment: 15000,
      msy: 25000,
      confidence_interval: [22000, 28000],
      recommendations: [
        'Continue current management practices',
        'Monitor recruitment trends',
        'Update stock assessments annually'
      ],
      last_updated: '2024-09-22'
    }
  ], []);

  // Chart components
  const ForecastChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={mockForecastData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <RechartsTooltip 
          labelFormatter={(value) => `Date: ${value}`}
          formatter={(value: any, name: string) => [
            typeof value === 'number' ? value.toFixed(2) : value,
            name === 'value' ? (mockForecastData.find(d => d.value === value)?.predicted ? 'Predicted' : 'Historical') : name
          ]}
        />
        <Legend />
        
        {/* Historical data */}
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#1890ff" 
          strokeWidth={2}
          dot={false}
          connectNulls={false}
        />
        
        {/* Confidence intervals for forecast */}
        <Area
          type="monotone"
          dataKey="upper_bound"
          stackId="1"
          stroke="none"
          fill="#91d5ff"
          fillOpacity={0.3}
        />
        <Area
          type="monotone"
          dataKey="lower_bound"
          stackId="1"
          stroke="none"
          fill="#ffffff"
          fillOpacity={1}
        />
        
        {/* Forecast line */}
        <Line
          type="monotone"
          dataKey="value"
          stroke="#ff4d4f"
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={{ fill: '#ff4d4f', r: 3 }}
        />
        
        <ReferenceLine x={dayjs().format('YYYY-MM')} stroke="green" strokeDasharray="2 2" />
        <Brush />
      </ComposedChart>
    </ResponsiveContainer>
  );

  const StockStatusChart = () => {
    const chartData = mockStockStatus.map(stock => ({
      species: stock.species,
      biomass: stock.biomass,
      msy: stock.msy,
      fishing_mortality: stock.fishing_mortality * 100,
      status: stock.status
    }));

    return (
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="species" />
          <YAxis />
          <RechartsTooltip />
          <Legend />
          <Bar dataKey="biomass" fill="#52c41a" name="Current Biomass" />
          <Bar dataKey="msy" fill="#1890ff" name="MSY" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const handleRunAnalysis = useCallback(() => {
    setRunningAnalysis(true);
    
    // Simulate analysis
    setTimeout(() => {
      setRunningAnalysis(false);
      toast.success('Analysis completed successfully');
      refetchForecast();
      refetchAssessments();
    }, 3000);
  }, [refetchForecast, refetchAssessments]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'green';
      case 'overfished': return 'red';
      case 'overfishing': return 'orange';
      default: return 'gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <TrendingUpOutlined />;
      case 'overfished': 
      case 'overfishing': return <WarningOutlined />;
      default: return null;
    }
  };

  if (modelsError) {
    return (
      <Alert
        message="Error loading predictive models"
        description="Failed to load available models. Please check your connection and try again."
        type="error"
        showIcon
        action={
          <Button size="small" onClick={() => window.location.reload()}>
            Retry
          </Button>
        }
      />
    );
  }

  return (
    <div className={`predictive-dashboard ${className}`}>
      {/* Dashboard Header */}
      <Card className="dashboard-header mb-4" size="small">
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              📊 Predictive Analytics Dashboard
            </Title>
            <Text type="secondary">
              Stock assessment, forecasting, and trend analysis for marine resources
            </Text>
          </Col>
          <Col>
            <Space>
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                loading={runningAnalysis}
                onClick={handleRunAnalysis}
              >
                {runningAnalysis ? 'Running...' : 'Run Analysis'}
              </Button>
              <Button icon={<SettingOutlined />} onClick={() => setModelConfigModal(true)}>
                Configure
              </Button>
              <Button icon={<DownloadOutlined />}>
                Export Results
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Control Panel */}
      <Card className="control-panel mb-4" size="small">
        <Row gutter={16}>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>Species Selection</Text>
              <Select
                mode="multiple"
                style={{ width: '100%' }}
                placeholder="Select species"
                value={selectedSpecies}
                onChange={setSelectedSpecies}
                loading={modelsLoading}
              >
                <Option value="all">All Species</Option>
                <Option value="hilsa">Hilsa Shad</Option>
                <Option value="mackerel">Indian Mackerel</Option>
                <Option value="sardine">Oil Sardine</Option>
                <Option value="tuna">Yellowfin Tuna</Option>
              </Select>
            </Space>
          </Col>
          
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>Models</Text>
              <Select
                mode="multiple"
                style={{ width: '100%' }}
                placeholder="Select models"
                value={selectedModels}
                onChange={setSelectedModels}
              >
                <Option value="arima">ARIMA</Option>
                <Option value="prophet">Prophet</Option>
                <Option value="lstm">LSTM Neural Network</Option>
                <Option value="surplus_production">Surplus Production</Option>
              </Select>
            </Space>
          </Col>
          
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>Forecast Horizon (months)</Text>
              <Slider
                min={1}
                max={60}
                value={forecastHorizon}
                onChange={setForecastHorizon}
                marks={{ 12: '1Y', 24: '2Y', 36: '3Y', 60: '5Y' }}
              />
              <Text type="secondary">{forecastHorizon} months</Text>
            </Space>
          </Col>
          
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>Date Range</Text>
              <RangePicker
                style={{ width: '100%' }}
                value={dateRange}
                onChange={setDateRange}
                format="YYYY-MM"
                picker="month"
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Main Dashboard Content */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        {/* Forecasting Tab */}
        <TabPane tab="📈 Forecasting" key="forecasting">
          <Row gutter={16}>
            <Col span={18}>
              <Card title="Time Series Forecast" loading={forecastLoading}>
                <ForecastChart />
              </Card>
            </Col>
            
            <Col span={6}>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Card size="small">
                  <Statistic 
                    title="Forecast Accuracy (R²)" 
                    value={0.87} 
                    precision={3}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>
                
                <Card size="small">
                  <Statistic 
                    title="Mean Absolute Error" 
                    value={12.5} 
                    precision={1}
                    suffix="units"
                  />
                </Card>
                
                <Card size="small">
                  <Statistic 
                    title="Confidence Level" 
                    value={confidenceLevel * 100} 
                    precision={0}
                    suffix="%"
                  />
                </Card>

                <Card size="small" title="Model Performance">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>ARIMA</Text>
                      <Progress percent={87} size="small" status="active" />
                    </div>
                    <div>
                      <Text>Prophet</Text>
                      <Progress percent={82} size="small" />
                    </div>
                    <div>
                      <Text>LSTM</Text>
                      <Progress percent={91} size="small" status="success" />
                    </div>
                  </Space>
                </Card>
              </Space>
            </Col>
          </Row>
        </TabPane>

        {/* Stock Assessment Tab */}
        <TabPane tab="🐟 Stock Assessment" key="assessment">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="Stock Status Overview" loading={assessmentLoading}>
                <StockStatusChart />
              </Card>
            </Col>
            
            <Col span={8}>
              <Card title="Assessment Summary" size="small">
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  {mockStockStatus.map((stock, index) => (
                    <Card key={index} size="small" className="stock-card">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text strong>{stock.species}</Text>
                          <Tag color={getStatusColor(stock.status)} icon={getStatusIcon(stock.status)}>
                            {stock.status.toUpperCase()}
                          </Tag>
                        </div>
                        
                        <Row gutter={8}>
                          <Col span={12}>
                            <Statistic
                              title="Biomass"
                              value={stock.biomass}
                              precision={0}
                              suffix="tons"
                              valueStyle={{ fontSize: '14px' }}
                            />
                          </Col>
                          <Col span={12}>
                            <Statistic
                              title="F/Fmsy"
                              value={stock.fishing_mortality / 0.3}
                              precision={2}
                              valueStyle={{ 
                                fontSize: '14px',
                                color: stock.fishing_mortality > 0.3 ? '#ff4d4f' : '#52c41a'
                              }}
                            />
                          </Col>
                        </Row>
                      </Space>
                    </Card>
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>

          {/* Detailed Assessment Table */}
          <Card title="Detailed Assessment Results" className="mt-4">
            <Table
              dataSource={mockStockStatus}
              rowKey="species"
              columns={[
                {
                  title: 'Species',
                  dataIndex: 'species',
                  key: 'species'
                },
                {
                  title: 'Status',
                  dataIndex: 'status',
                  key: 'status',
                  render: (status: string) => (
                    <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
                      {status.toUpperCase()}
                    </Tag>
                  )
                },
                {
                  title: 'Biomass (tons)',
                  dataIndex: 'biomass',
                  key: 'biomass',
                  render: (value: number) => value.toLocaleString()
                },
                {
                  title: 'MSY (tons)',
                  dataIndex: 'msy',
                  key: 'msy',
                  render: (value: number) => value.toLocaleString()
                },
                {
                  title: 'Fishing Mortality',
                  dataIndex: 'fishing_mortality',
                  key: 'fishing_mortality',
                  render: (value: number) => value.toFixed(3)
                },
                {
                  title: 'Last Updated',
                  dataIndex: 'last_updated',
                  key: 'last_updated'
                }
              ]}
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        {/* Trend Analysis Tab */}
        <TabPane tab="📊 Trend Analysis" key="trends">
          <Card title="Environmental and Biological Trends" loading={trendAnalysisMutation.isLoading}>
            <Row gutter={16}>
              <Col span={12}>
                <Card size="small" title="Temperature Trends">
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={mockForecastData.slice(0, 24)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <RechartsTooltip />
                      <Line type="monotone" dataKey="value" stroke="#ff7300" />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card size="small" title="Catch Per Unit Effort">
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={mockForecastData.slice(0, 24)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <RechartsTooltip />
                      <Area type="monotone" dataKey="value" stroke="#82ca9d" fill="#82ca9d" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>
      </Tabs>

      {/* Model Configuration Modal */}
      <Modal
        title="Configure Predictive Models"
        visible={modelConfigModal}
        onCancel={() => setModelConfigModal(false)}
        footer={null}
        width={600}
      >
        <Form
          onFinish={(values) => {
            trainModelMutation.mutate(values);
          }}
          layout="vertical"
        >
          <Form.Item label="Model Type" name="model_type" rules={[{ required: true }]}>
            <Select placeholder="Select model type">
              <Option value="arima">ARIMA</Option>
              <Option value="prophet">Prophet</Option>
              <Option value="lstm">LSTM</Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="Training Data Split" name="validation_split">
            <Slider min={0.1} max={0.5} step={0.1} defaultValue={0.2} marks={{ 0.2: '20%', 0.3: '30%' }} />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="Seasonality Periods" name="seasonality_periods">
                <InputNumber min={1} max={365} defaultValue={12} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="Max Iterations" name="max_iterations">
                <InputNumber min={100} max={10000} defaultValue={1000} />
              </Form.Item>
            </Col>
          </Row>
          
          <Space>
            <Button type="primary" htmlType="submit" loading={trainModelMutation.isLoading}>
              Start Training
            </Button>
            <Button onClick={() => setModelConfigModal(false)}>
              Cancel
            </Button>
          </Space>
        </Form>
      </Modal>

      <style jsx>{`
        .predictive-dashboard {
          padding: 0;
        }
        
        .stock-card {
          border: 1px solid #f0f0f0;
          border-radius: 6px;
        }
        
        .stock-card .ant-card-body {
          padding: 12px;
        }
      `}</style>
    </div>
  );
};

export default PredictiveDashboard;